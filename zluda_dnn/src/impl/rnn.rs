// Implementation of the cuDNN 8 legacy RNN API (cudnnSetRNNDescriptor_v6, cudnnRNN*Ex, ...)
// on top of the MIOpen sequence tensor RNN API.
//
// The main difference between the two is the layout of the packed weight buffer. cuDNN stores
// all the weight matrices (for each pseudo-layer: the input matrices, then the recurrent
// matrices) followed by all the biases in the same order. Frameworks (e.g. MXNet) fill the buffer
// according to that layout. MIOpen uses a different layout and a different gate order. Every
// call converts the weights to the MIOpen layout in a part of the workspace, and the weight
// gradients back to the cuDNN layout.
//
// Workspace layout: [weights in MIOpen layout | weight gradients | MIOpen workspace]

use super::{miopen, Context, MIOpenVtable};
use cuda_types::cudnn9::*;
use hip_runtime_sys::*;
use miopen_sys::*;
use std::{
    ffi::c_void,
    mem, ptr,
    sync::{atomic::Ordering, Arc, Mutex},
};

const RNN_COOKIE: usize = 0x4c1d_8e2a_93f5_0b67;
const RNN_DATA_COOKIE: usize = 0x7a0e_36c4_d915_2f88;
const WORKSPACE_ALIGNMENT: usize = 256;

pub(crate) struct RnnDescriptor {
    cookie: usize,
    base: miopenRNNDescriptor_t,
    state: Mutex<RnnState>,
}

#[derive(Default)]
struct RnnState {
    config: Option<RnnConfig>,
    // Cached weight layout for the last used input size and data type
    layout: Option<Arc<WeightLayout>>,
}

#[derive(Clone, Copy)]
struct RnnConfig {
    mode: cudnnRNNMode_t,
    hidden_size: usize,
    layers: usize,
    directions: usize,
}

impl RnnConfig {
    fn gates(&self) -> usize {
        match self.mode {
            cudnnRNNMode_t::CUDNN_LSTM => 4,
            cudnnRNNMode_t::CUDNN_GRU => 3,
            _ => 1,
        }
    }

    // MIOpen index of the cuDNN linear layer `lin_id` of a pseudo-layer.
    // cuDNN LSTM gate order: input, forget, new memory, output
    // MIOpen LSTM gate order: input, forget, output, new memory
    // cuDNN GRU gate order: reset, update, new memory
    // MIOpen GRU gate order: update, reset, new memory
    fn miopen_id(&self, lin_id: usize) -> usize {
        let gates = self.gates();
        let (recurrent, gate) = (lin_id / gates, lin_id % gates);
        let gate = match self.mode {
            cudnnRNNMode_t::CUDNN_LSTM => [0, 1, 3, 2][gate],
            cudnnRNNMode_t::CUDNN_GRU => [1, 0, 2][gate],
            _ => gate,
        };
        recurrent * gates + gate
    }
}

// A contiguous range of elements, which has the offset `cudnn` in the cuDNN weight buffer and the
// offset `miopen` in the MIOpen weight buffer
#[derive(Clone, Copy, Debug, PartialEq)]
struct Block {
    cudnn: usize,
    miopen: usize,
    len: usize,
}

struct WeightLayout {
    input_size: usize,
    data_type: miopenDataType_t,
    elements: usize,
    blocks: Vec<Block>,
}

impl WeightLayout {
    fn bytes(&self) -> Result<usize, cudnnError_t> {
        Ok(self.elements * element_size(self.data_type)?)
    }

    // Size of the part of the workspace in front of the MIOpen workspace
    fn workspace_prefix(&self) -> Result<usize, cudnnError_t> {
        Ok(2 * self.bytes()?.next_multiple_of(WORKSPACE_ALIGNMENT))
    }
}

pub(crate) struct RnnDataDescriptor {
    cookie: usize,
    base: miopenSeqTensorDescriptor_t,
    info: Mutex<Option<RnnDataInfo>>,
}

#[derive(Clone)]
struct RnnDataInfo {
    data_type: miopenDataType_t,
    // None if the data is packed
    padded: Option<PaddedLayout>,
    max_seq_length: usize,
    batch_size: usize,
    vector_size: usize,
    seq_lengths: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq)]
enum PaddedLayout {
    SeqMajor,
    BatchMajor,
}

fn element_size(data_type: miopenDataType_t) -> Result<usize, cudnnError_t> {
    Ok(match data_type {
        miopenDataType_t::miopenHalf | miopenDataType_t::miopenBFloat16 => 2,
        miopenDataType_t::miopenFloat => 4,
        miopenDataType_t::miopenDouble => 8,
        _ => return Err(cudnnError_t::NOT_SUPPORTED),
    })
}

unsafe fn as_rnn<'a>(desc: cudnnRNNDescriptor_t) -> Result<&'a RnnDescriptor, cudnnError_t> {
    let desc = desc
        .cast::<RnnDescriptor>()
        .as_ref()
        .ok_or(cudnnError_t::BAD_PARAM)?;
    if desc.cookie != RNN_COOKIE {
        return Err(cudnnError_t::BAD_PARAM);
    }
    Ok(desc)
}

unsafe fn as_rnn_data<'a>(
    desc: cudnnRNNDataDescriptor_t,
) -> Result<&'a RnnDataDescriptor, cudnnError_t> {
    let desc = desc
        .cast::<RnnDataDescriptor>()
        .as_ref()
        .ok_or(cudnnError_t::BAD_PARAM)?;
    if desc.cookie != RNN_DATA_COOKIE {
        return Err(cudnnError_t::BAD_PARAM);
    }
    Ok(desc)
}

impl RnnDescriptor {
    fn config(&self) -> Result<RnnConfig, cudnnError_t> {
        self.state
            .lock()
            .map_err(|_| cudnnError_t::INTERNAL_ERROR)?
            .config
            .ok_or(cudnnError_t::BAD_PARAM)
    }

    unsafe fn weight_layout(
        &self,
        miopen: &MIOpenVtable,
        input_size: usize,
        data_type: miopenDataType_t,
    ) -> Result<Arc<WeightLayout>, cudnnError_t> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| cudnnError_t::INTERNAL_ERROR)?;
        let config = state.config.ok_or(cudnnError_t::BAD_PARAM)?;
        if let Some(layout) = &state.layout {
            if layout.input_size == input_size && layout.data_type == data_type {
                return Ok(layout.clone());
            }
        }
        let layout = Arc::new(compute_weight_layout(
            miopen, self.base, config, input_size, data_type,
        )?);
        state.layout = Some(layout.clone());
        Ok(layout)
    }
}

struct TensorDescriptor<'a>(&'a MIOpenVtable, miopenTensorDescriptor_t);

impl<'a> TensorDescriptor<'a> {
    unsafe fn new(
        miopen: &'a MIOpenVtable,
        data_type: miopenDataType_t,
        dims: &[i32],
    ) -> Result<Self, cudnnError_t> {
        let mut desc = mem::zeroed();
        miopen.miopenCreateTensorDescriptor(&mut desc)?;
        let desc = Self(miopen, desc);
        let mut strides = vec![1i32; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        miopen.miopenSetTensorDescriptor(
            desc.1,
            data_type,
            dims.len() as i32,
            dims.as_ptr(),
            strides.as_ptr(),
        )?;
        Ok(desc)
    }
}

impl Drop for TensorDescriptor<'_> {
    fn drop(&mut self) {
        unsafe { self.0.miopenDestroyTensorDescriptor(self.1) }.ok();
    }
}

struct SeqTensorDescriptor<'a>(&'a MIOpenVtable, miopenSeqTensorDescriptor_t);

impl<'a> SeqTensorDescriptor<'a> {
    unsafe fn new(
        miopen: &'a MIOpenVtable,
        data_type: miopenDataType_t,
        layout: miopenRNNBaseLayout_t,
        max_seq_length: usize,
        batch_size: usize,
        vector_size: usize,
    ) -> Result<Self, cudnnError_t> {
        let mut desc = mem::zeroed();
        miopen.miopenCreateSeqTensorDescriptor(&mut desc)?;
        let desc = Self(miopen, desc);
        let lengths = vec![max_seq_length as i32; batch_size];
        miopen.miopenSetRNNDataSeqTensorDescriptor(
            desc.1,
            data_type,
            layout,
            max_seq_length as i32,
            batch_size as i32,
            vector_size as i32,
            lengths.as_ptr(),
            ptr::null_mut(),
        )?;
        Ok(desc)
    }
}

impl Drop for SeqTensorDescriptor<'_> {
    fn drop(&mut self) {
        unsafe { self.0.miopenDestroySeqTensorDescriptor(self.1) }.ok();
    }
}

unsafe fn compute_weight_layout(
    miopen: &MIOpenVtable,
    rnn: miopenRNNDescriptor_t,
    config: RnnConfig,
    input_size: usize,
    data_type: miopenDataType_t,
) -> Result<WeightLayout, cudnnError_t> {
    let x_desc = TensorDescriptor::new(miopen, data_type, &[1, input_size as i32])?;
    let param_desc = TensorDescriptor::new(miopen, data_type, &[1])?;
    let gates = config.gates();
    let hidden = config.hidden_size;
    let pseudo_layers = config.layers * config.directions;
    let element_size = element_size(data_type)?;
    let mut blocks: Vec<Block> = Vec::new();
    let mut cudnn_offset = 0;
    let mut add_block = |miopen_offset: usize, len: usize, desc_bytes: usize| {
        if desc_bytes != len * element_size {
            return Err(cudnnError_t::NOT_SUPPORTED);
        }
        let block = Block {
            cudnn: cudnn_offset,
            miopen: miopen_offset,
            len,
        };
        cudnn_offset += len;
        match blocks.last_mut() {
            Some(last)
                if last.cudnn + last.len == block.cudnn
                    && last.miopen + last.len == block.miopen =>
            {
                last.len += block.len
            }
            _ => blocks.push(block),
        }
        Ok(())
    };
    // Weight matrices
    for pseudo_layer in 0..pseudo_layers {
        let layer_input = if pseudo_layer < config.directions {
            input_size
        } else {
            hidden * config.directions
        };
        for lin_id in 0..2 * gates {
            let len = hidden * if lin_id < gates { layer_input } else { hidden };
            let mut offset = 0;
            miopen.miopenGetRNNLayerParamOffset(
                rnn,
                pseudo_layer as i32,
                x_desc.1,
                config.miopen_id(lin_id) as i32,
                param_desc.1,
                &mut offset,
            )?;
            let mut desc_bytes = 0;
            miopen.miopenGetTensorNumBytes(param_desc.1, &mut desc_bytes)?;
            add_block(offset, len, desc_bytes)?;
        }
    }
    // Biases
    for pseudo_layer in 0..pseudo_layers {
        for lin_id in 0..2 * gates {
            let mut offset = 0;
            miopen.miopenGetRNNLayerBiasOffset(
                rnn,
                pseudo_layer as i32,
                x_desc.1,
                config.miopen_id(lin_id) as i32,
                param_desc.1,
                &mut offset,
            )?;
            let mut desc_bytes = 0;
            miopen.miopenGetTensorNumBytes(param_desc.1, &mut desc_bytes)?;
            add_block(offset, hidden, desc_bytes)?;
        }
    }
    let elements = cudnn_offset;
    // Both layouts must contain the same elements
    let mut miopen_ranges = blocks
        .iter()
        .map(|b| (b.miopen, b.miopen + b.len))
        .collect::<Vec<_>>();
    miopen_ranges.sort_unstable();
    let mut end = 0;
    for (start, range_end) in miopen_ranges {
        if start != end {
            return Err(cudnnError_t::NOT_SUPPORTED);
        }
        end = range_end;
    }
    if end != elements {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    Ok(WeightLayout {
        input_size,
        data_type,
        elements,
        blocks,
    })
}

unsafe fn convert_weights(
    layout: &WeightLayout,
    src: *const c_void,
    dst: *mut c_void,
    to_miopen: bool,
    stream: hipStream_t,
) -> Result<(), cudnnError_t> {
    let element_size = element_size(layout.data_type)?;
    for block in layout.blocks.iter() {
        let (src_offset, dst_offset) = if to_miopen {
            (block.cudnn, block.miopen)
        } else {
            (block.miopen, block.cudnn)
        };
        hipMemcpyDtoDAsync(
            hipDeviceptr_t(dst.cast::<u8>().add(dst_offset * element_size).cast()),
            hipDeviceptr_t(
                src.cast::<u8>()
                    .add(src_offset * element_size)
                    .cast_mut()
                    .cast(),
            ),
            block.len * element_size,
            stream,
        )
        .map_err(|_| cudnnError_t::EXECUTION_FAILED)?;
    }
    Ok(())
}

fn stream(handle: &Context) -> hipStream_t {
    hipStream_t(handle.stream.load(Ordering::Acquire))
}

pub(crate) unsafe fn create_rnn_descriptor(rnn_desc: &mut cudnnRNNDescriptor_t) -> cudnnStatus_t {
    let mut base = mem::zeroed();
    miopen()?.miopenCreateRNNDescriptor(&mut base)?;
    *rnn_desc = Box::into_raw(Box::new(RnnDescriptor {
        cookie: RNN_COOKIE,
        base,
        state: Mutex::new(RnnState::default()),
    }))
    .cast();
    Ok(())
}

pub(crate) unsafe fn destroy_rnn_descriptor(rnn_desc: cudnnRNNDescriptor_t) -> cudnnStatus_t {
    as_rnn(rnn_desc)?;
    let mut desc = Box::from_raw(rnn_desc.cast::<RnnDescriptor>());
    desc.cookie = 0;
    miopen()?.miopenDestroyRNNDescriptor(desc.base)?;
    Ok(())
}

unsafe fn dropout_rate(
    miopen: &MIOpenVtable,
    handle: &Context,
    dropout_desc: miopenDropoutDescriptor_t,
) -> Result<f32, cudnnError_t> {
    let mut dropout = 0.0;
    let mut states = ptr::null_mut();
    let mut seed = 0;
    let mut use_mask = false;
    let mut state_evo = false;
    let mut rng_mode = mem::zeroed();
    miopen.miopenGetDropoutDescriptor(
        dropout_desc,
        handle.base,
        &mut dropout,
        &mut states,
        &mut seed,
        &mut use_mask,
        &mut state_evo,
        &mut rng_mode,
    )?;
    Ok(dropout)
}

pub(crate) unsafe fn set_rnn_descriptor_v6(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    hidden_size: i32,
    num_layers: i32,
    dropout_desc: miopenDropoutDescriptor_t,
    input_mode: cudnnRNNInputMode_t,
    direction: cudnnDirectionMode_t,
    cell_mode: cudnnRNNMode_t,
    _algo: cudnnRNNAlgo_t,
    math_prec: miopenDataType_t,
) -> cudnnStatus_t {
    let desc = as_rnn(rnn_desc)?;
    if hidden_size <= 0 || num_layers <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    // In skip mode the first layer has no input matrices, which changes the weight layout
    if input_mode != cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    let (miopen_direction, directions) = match direction {
        cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL => {
            (miopenRNNDirectionMode_t::miopenRNNunidirection, 1)
        }
        cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL => {
            (miopenRNNDirectionMode_t::miopenRNNbidirection, 2)
        }
        _ => return Err(cudnnError_t::BAD_PARAM),
    };
    let miopen_mode = match cell_mode {
        cudnnRNNMode_t::CUDNN_RNN_RELU => miopenRNNMode_t::miopenRNNRELU,
        cudnnRNNMode_t::CUDNN_RNN_TANH => miopenRNNMode_t::miopenRNNTANH,
        cudnnRNNMode_t::CUDNN_LSTM => miopenRNNMode_t::miopenLSTM,
        cudnnRNNMode_t::CUDNN_GRU => miopenRNNMode_t::miopenGRU,
        _ => return Err(cudnnError_t::BAD_PARAM),
    };
    let miopen = miopen()?;
    // Without dropout, MIOpen doesn't need the descriptor (which may have no random number
    // generator states)
    if dropout_desc.0.is_null() || dropout_rate(miopen, handle, dropout_desc)? == 0.0 {
        miopen.miopenSetRNNDescriptor(
            desc.base,
            hidden_size,
            num_layers,
            miopenRNNInputMode_t::miopenRNNlinear,
            miopen_direction,
            miopen_mode,
            miopenRNNBiasMode_t::miopenRNNwithBias,
            miopenRNNAlgo_t::miopenRNNdefault,
            math_prec,
        )?;
    } else {
        miopen.miopenSetRNNDescriptor_V2(
            desc.base,
            hidden_size,
            num_layers,
            dropout_desc,
            miopenRNNInputMode_t::miopenRNNlinear,
            miopen_direction,
            miopen_mode,
            miopenRNNBiasMode_t::miopenRNNwithBias,
            miopenRNNAlgo_t::miopenRNNdefault,
            math_prec,
        )?;
    }
    let mut state = desc
        .state
        .lock()
        .map_err(|_| cudnnError_t::INTERNAL_ERROR)?;
    state.config = Some(RnnConfig {
        mode: cell_mode,
        hidden_size: hidden_size as usize,
        layers: num_layers as usize,
        directions,
    });
    state.layout = None;
    Ok(())
}

pub(crate) unsafe fn set_rnn_matrix_math_type(
    rnn_desc: cudnnRNNDescriptor_t,
    _m_type: cudnnMathType_t,
) -> cudnnStatus_t {
    as_rnn(rnn_desc)?;
    Ok(())
}

pub(crate) unsafe fn set_rnn_projection_layers(
    _handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    rec_proj_size: i32,
    out_proj_size: i32,
) -> cudnnStatus_t {
    let config = as_rnn(rnn_desc)?.config()?;
    // Only "no projection" is supported
    if rec_proj_size as usize == config.hidden_size && out_proj_size == 0 {
        Ok(())
    } else {
        Err(cudnnError_t::NOT_SUPPORTED)
    }
}

// The padding is described by the RNN data descriptors
pub(crate) unsafe fn set_rnn_padding_mode(
    rnn_desc: cudnnRNNDescriptor_t,
    _padding_mode: u32,
) -> cudnnStatus_t {
    as_rnn(rnn_desc)?;
    Ok(())
}

pub(crate) unsafe fn rnn_set_clip(
    _handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    clip_mode: cudnnRNNClipMode_t,
    _clip_nan_opt: cudnnNanPropagation_t,
    _lclip: f64,
    _rclip: f64,
) -> cudnnStatus_t {
    as_rnn(rnn_desc)?;
    if clip_mode == cudnnRNNClipMode_t::CUDNN_RNN_CLIP_NONE {
        Ok(())
    } else {
        Err(cudnnError_t::NOT_SUPPORTED)
    }
}

// Batch size, input size and data type of a per time step tensor descriptor
unsafe fn step_tensor_info(
    miopen: &MIOpenVtable,
    desc: miopenTensorDescriptor_t,
) -> Result<(usize, usize, miopenDataType_t), cudnnError_t> {
    let mut data_type = mem::zeroed();
    let mut dims = [0i32; 8];
    let mut strides = [0i32; 8];
    let mut nb_dims = 0;
    miopen.miopenGetTensorDescriptorSize(desc, &mut nb_dims)?;
    if !(2..=8).contains(&nb_dims) {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen.miopenGetTensorDescriptor(
        desc,
        &mut data_type,
        dims.as_mut_ptr(),
        strides.as_mut_ptr(),
    )?;
    Ok((dims[0] as usize, dims[1] as usize, data_type))
}

pub(crate) unsafe fn get_rnn_params_size(
    _handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    x_desc: miopenTensorDescriptor_t,
    size_in_bytes: &mut usize,
    data_type: miopenDataType_t,
) -> cudnnStatus_t {
    let miopen = miopen()?;
    let (_, input_size, _) = step_tensor_info(miopen, x_desc)?;
    let layout = as_rnn(rnn_desc)?.weight_layout(miopen, input_size, data_type)?;
    *size_in_bytes = layout.bytes()?;
    Ok(())
}

// Maximum MIOpen workspace and reserve space sizes over the supported data layouts and modes
unsafe fn miopen_temp_space_sizes(
    miopen: &MIOpenVtable,
    handle: &Context,
    rnn: miopenRNNDescriptor_t,
    data_type: miopenDataType_t,
    seq_length: usize,
    batch_size: usize,
    input_size: usize,
) -> Result<(usize, usize), cudnnError_t> {
    let mut workspace = 0;
    let mut reserve = 0;
    for layout in [
        miopenRNNBaseLayout_t::miopenRNNDataSeqMajorNotPadded,
        miopenRNNBaseLayout_t::miopenRNNDataSeqMajorPadded,
    ] {
        let x_desc = SeqTensorDescriptor::new(
            miopen, data_type, layout, seq_length, batch_size, input_size,
        )?;
        for mode in [
            miopenRNNFWDMode_t::miopenRNNTraining,
            miopenRNNFWDMode_t::miopenRNNInference,
        ] {
            let mut mode_workspace = 0;
            let mut mode_reserve = 0;
            miopen.miopenGetRNNTempSpaceSizes(
                handle.base,
                rnn,
                x_desc.1,
                mode,
                &mut mode_workspace,
                &mut mode_reserve,
            )?;
            workspace = workspace.max(mode_workspace);
            reserve = reserve.max(mode_reserve);
        }
    }
    Ok((workspace, reserve))
}

unsafe fn temp_space_sizes(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    seq_length: i32,
    x_desc: miopenTensorDescriptor_t,
) -> Result<(usize, usize), cudnnError_t> {
    if seq_length <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let miopen = miopen()?;
    let rnn = as_rnn(rnn_desc)?;
    let (batch_size, input_size, data_type) = step_tensor_info(miopen, x_desc)?;
    let layout = rnn.weight_layout(miopen, input_size, data_type)?;
    let (workspace, reserve) = miopen_temp_space_sizes(
        miopen,
        handle,
        rnn.base,
        data_type,
        seq_length as usize,
        batch_size,
        input_size,
    )?;
    Ok((layout.workspace_prefix()? + workspace, reserve))
}

pub(crate) unsafe fn get_rnn_workspace_size(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    seq_length: i32,
    x_desc: &miopenTensorDescriptor_t,
    size_in_bytes: &mut usize,
) -> cudnnStatus_t {
    *size_in_bytes = temp_space_sizes(handle, rnn_desc, seq_length, *x_desc)?.0;
    Ok(())
}

pub(crate) unsafe fn get_rnn_training_reserve_size(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    seq_length: i32,
    x_desc: &miopenTensorDescriptor_t,
    size_in_bytes: &mut usize,
) -> cudnnStatus_t {
    *size_in_bytes = temp_space_sizes(handle, rnn_desc, seq_length, *x_desc)?.1;
    Ok(())
}

pub(crate) unsafe fn create_rnn_data_descriptor(
    rnn_data_desc: &mut cudnnRNNDataDescriptor_t,
) -> cudnnStatus_t {
    let mut base = mem::zeroed();
    miopen()?.miopenCreateSeqTensorDescriptor(&mut base)?;
    *rnn_data_desc = Box::into_raw(Box::new(RnnDataDescriptor {
        cookie: RNN_DATA_COOKIE,
        base,
        info: Mutex::new(None),
    }))
    .cast();
    Ok(())
}

pub(crate) unsafe fn destroy_rnn_data_descriptor(
    rnn_data_desc: cudnnRNNDataDescriptor_t,
) -> cudnnStatus_t {
    as_rnn_data(rnn_data_desc)?;
    let mut desc = Box::from_raw(rnn_data_desc.cast::<RnnDataDescriptor>());
    desc.cookie = 0;
    miopen()?.miopenDestroySeqTensorDescriptor(desc.base)?;
    Ok(())
}

pub(crate) unsafe fn set_rnn_data_descriptor(
    rnn_data_desc: cudnnRNNDataDescriptor_t,
    data_type: miopenDataType_t,
    layout: cudnnRNNDataLayout_t,
    max_seq_length: i32,
    batch_size: i32,
    vector_size: i32,
    seq_length_array: *const i32,
    padding_fill: *mut c_void,
) -> cudnnStatus_t {
    let desc = as_rnn_data(rnn_data_desc)?;
    if seq_length_array.is_null() || max_seq_length <= 0 || batch_size <= 0 || vector_size <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let (miopen_layout, padded) = match layout {
        cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED => {
            (miopenRNNBaseLayout_t::miopenRNNDataSeqMajorNotPadded, None)
        }
        cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED => (
            miopenRNNBaseLayout_t::miopenRNNDataSeqMajorPadded,
            Some(PaddedLayout::SeqMajor),
        ),
        cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED => (
            miopenRNNBaseLayout_t::miopenRNNDataBatchMajorPadded,
            Some(PaddedLayout::BatchMajor),
        ),
        _ => return Err(cudnnError_t::BAD_PARAM),
    };
    let seq_lengths = std::slice::from_raw_parts(seq_length_array, batch_size as usize)
        .iter()
        .map(|length| {
            if (0..=max_seq_length).contains(length) {
                Ok(*length as usize)
            } else {
                Err(cudnnError_t::BAD_PARAM)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Padding in outputs is filled with zeros (see `zero_padding`)
    if padded.is_some() && !padding_fill.is_null() {
        let fill = std::slice::from_raw_parts(padding_fill.cast::<u8>(), element_size(data_type)?);
        if fill.iter().any(|b| *b != 0) {
            return Err(cudnnError_t::NOT_SUPPORTED);
        }
    }
    miopen()?.miopenSetRNNDataSeqTensorDescriptor(
        desc.base,
        data_type,
        miopen_layout,
        max_seq_length,
        batch_size,
        vector_size,
        seq_length_array,
        ptr::null_mut(),
    )?;
    *desc.info.lock().map_err(|_| cudnnError_t::INTERNAL_ERROR)? = Some(RnnDataInfo {
        data_type,
        padded,
        max_seq_length: max_seq_length as usize,
        batch_size: batch_size as usize,
        vector_size: vector_size as usize,
        seq_lengths,
    });
    Ok(())
}

unsafe fn rnn_data<'a>(
    desc: cudnnRNNDataDescriptor_t,
) -> Result<(&'a RnnDataDescriptor, RnnDataInfo), cudnnError_t> {
    let desc = as_rnn_data(desc)?;
    let info = desc
        .info
        .lock()
        .map_err(|_| cudnnError_t::INTERNAL_ERROR)?
        .clone()
        .ok_or(cudnnError_t::BAD_PARAM)?;
    Ok((desc, info))
}

// cuDNN fills the padding of output sequences with zeros, MIOpen leaves values there
unsafe fn zero_padding(
    info: &RnnDataInfo,
    data: *mut c_void,
    stream: hipStream_t,
) -> Result<(), cudnnError_t> {
    let Some(layout) = info.padded else {
        return Ok(());
    };
    if data.is_null() {
        return Ok(());
    }
    let element_size = element_size(info.data_type)?;
    let vector_bytes = info.vector_size * element_size;
    for (batch, length) in info.seq_lengths.iter().copied().enumerate() {
        let padding_steps = info.max_seq_length - length;
        if padding_steps == 0 {
            continue;
        }
        let result = match layout {
            // Padding of one sequence is strided
            PaddedLayout::SeqMajor => hipMemset2DAsync(
                data.cast::<u8>()
                    .add((length * info.batch_size + batch) * vector_bytes)
                    .cast(),
                info.batch_size * vector_bytes,
                0,
                vector_bytes,
                padding_steps,
                stream,
            ),
            // Padding of one sequence is contiguous
            PaddedLayout::BatchMajor => hipMemsetAsync(
                data.cast::<u8>()
                    .add((batch * info.max_seq_length + length) * vector_bytes)
                    .cast(),
                0,
                padding_steps * vector_bytes,
                stream,
            ),
        };
        result.map_err(|_| cudnnError_t::EXECUTION_FAILED)?;
    }
    Ok(())
}

struct Workspace {
    // Space for a weight buffer in the MIOpen layout
    weights: *mut c_void,
    // Space for a weight buffer in the cuDNN layout
    weights_cudnn: *mut c_void,
    miopen: *mut c_void,
    miopen_size: usize,
}

unsafe fn split_workspace(
    layout: &WeightLayout,
    workspace: *mut c_void,
    workspace_size: usize,
) -> Result<Workspace, cudnnError_t> {
    let prefix = layout.workspace_prefix()?;
    if workspace.is_null() || workspace_size < prefix {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let workspace = workspace.cast::<u8>();
    Ok(Workspace {
        weights: workspace.cast(),
        weights_cudnn: workspace.add(prefix / 2).cast(),
        miopen: workspace.add(prefix).cast(),
        miopen_size: workspace_size - prefix,
    })
}

fn check_no_attention(pointers: &[*const c_void]) -> Result<(), cudnnError_t> {
    if pointers.iter().all(|p| p.is_null()) {
        Ok(())
    } else {
        Err(cudnnError_t::NOT_SUPPORTED)
    }
}

unsafe fn rnn_forward(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    mode: miopenRNNFWDMode_t,
    x_desc: cudnnRNNDataDescriptor_t,
    x: *const c_void,
    hx_desc: miopenTensorDescriptor_t,
    hx: *const c_void,
    cx_desc: miopenTensorDescriptor_t,
    cx: *const c_void,
    w: *const c_void,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *mut c_void,
    hy: *mut c_void,
    cy: *mut c_void,
    workspace: *mut c_void,
    workspace_size: usize,
    reserve_space: *mut c_void,
    reserve_space_size: usize,
) -> cudnnStatus_t {
    let miopen = miopen()?;
    let rnn = as_rnn(rnn_desc)?;
    let (x_desc, x_info) = rnn_data(x_desc)?;
    let (y_desc, y_info) = rnn_data(y_desc)?;
    let layout = rnn.weight_layout(miopen, x_info.vector_size, x_info.data_type)?;
    let workspace = split_workspace(&layout, workspace, workspace_size)?;
    let stream = stream(handle);
    convert_weights(&layout, w, workspace.weights, true, stream)?;
    miopen.miopenRNNForward(
        handle.base,
        rnn.base,
        mode,
        x_desc.base,
        x,
        hx_desc,
        hx,
        hy,
        cx_desc,
        cx,
        cy,
        y_desc.base,
        y,
        workspace.weights,
        layout.bytes()?,
        workspace.miopen,
        workspace.miopen_size,
        reserve_space,
        reserve_space_size,
    )?;
    zero_padding(&y_info, y, stream)
}

pub(crate) unsafe fn rnn_forward_training_ex(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    x_desc: cudnnRNNDataDescriptor_t,
    x: *const c_void,
    hx_desc: miopenTensorDescriptor_t,
    hx: *const c_void,
    cx_desc: miopenTensorDescriptor_t,
    cx: *const c_void,
    _w_desc: miopenTensorDescriptor_t,
    w: *const c_void,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *mut c_void,
    _hy_desc: miopenTensorDescriptor_t,
    hy: *mut c_void,
    _cy_desc: miopenTensorDescriptor_t,
    cy: *mut c_void,
    _k_desc: cudnnRNNDataDescriptor_t,
    keys: *const c_void,
    _c_desc: cudnnRNNDataDescriptor_t,
    c_attn: *mut c_void,
    _i_desc: cudnnRNNDataDescriptor_t,
    i_attn: *mut c_void,
    _q_desc: cudnnRNNDataDescriptor_t,
    queries: *mut c_void,
    workspace: *mut c_void,
    workspace_size_in_bytes: usize,
    reserve_space: *mut c_void,
    reserve_space_size_in_bytes: usize,
) -> cudnnStatus_t {
    check_no_attention(&[
        keys,
        c_attn.cast_const(),
        i_attn.cast_const(),
        queries.cast_const(),
    ])?;
    rnn_forward(
        handle,
        rnn_desc,
        miopenRNNFWDMode_t::miopenRNNTraining,
        x_desc,
        x,
        hx_desc,
        hx,
        cx_desc,
        cx,
        w,
        y_desc,
        y,
        hy,
        cy,
        workspace,
        workspace_size_in_bytes,
        reserve_space,
        reserve_space_size_in_bytes,
    )
}

pub(crate) unsafe fn rnn_forward_inference_ex(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    x_desc: cudnnRNNDataDescriptor_t,
    x: *const c_void,
    hx_desc: miopenTensorDescriptor_t,
    hx: *const c_void,
    cx_desc: miopenTensorDescriptor_t,
    cx: *const c_void,
    _w_desc: miopenTensorDescriptor_t,
    w: *const c_void,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *mut c_void,
    _hy_desc: miopenTensorDescriptor_t,
    hy: *mut c_void,
    _cy_desc: miopenTensorDescriptor_t,
    cy: *mut c_void,
    _k_desc: cudnnRNNDataDescriptor_t,
    keys: *const c_void,
    _c_desc: cudnnRNNDataDescriptor_t,
    c_attn: *mut c_void,
    _i_desc: cudnnRNNDataDescriptor_t,
    i_attn: *mut c_void,
    _q_desc: cudnnRNNDataDescriptor_t,
    queries: *mut c_void,
    workspace: *mut c_void,
    workspace_size_in_bytes: usize,
) -> cudnnStatus_t {
    check_no_attention(&[
        keys,
        c_attn.cast_const(),
        i_attn.cast_const(),
        queries.cast_const(),
    ])?;
    rnn_forward(
        handle,
        rnn_desc,
        miopenRNNFWDMode_t::miopenRNNInference,
        x_desc,
        x,
        hx_desc,
        hx,
        cx_desc,
        cx,
        w,
        y_desc,
        y,
        hy,
        cy,
        workspace,
        workspace_size_in_bytes,
        ptr::null_mut(),
        0,
    )
}

pub(crate) unsafe fn rnn_backward_data_ex(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *const c_void,
    _dy_desc: cudnnRNNDataDescriptor_t,
    dy: *const c_void,
    _dc_desc: cudnnRNNDataDescriptor_t,
    dc_attn: *const c_void,
    _dhy_desc: miopenTensorDescriptor_t,
    dhy: *const c_void,
    _dcy_desc: miopenTensorDescriptor_t,
    dcy: *const c_void,
    _w_desc: miopenTensorDescriptor_t,
    w: *const c_void,
    hx_desc: miopenTensorDescriptor_t,
    hx: *const c_void,
    cx_desc: miopenTensorDescriptor_t,
    cx: *const c_void,
    dx_desc: cudnnRNNDataDescriptor_t,
    dx: *mut c_void,
    _dhx_desc: miopenTensorDescriptor_t,
    dhx: *mut c_void,
    _dcx_desc: miopenTensorDescriptor_t,
    dcx: *mut c_void,
    _dk_desc: cudnnRNNDataDescriptor_t,
    dkeys: *mut c_void,
    workspace: *mut c_void,
    workspace_size_in_bytes: usize,
    reserve_space: *mut c_void,
    reserve_space_size_in_bytes: usize,
) -> cudnnStatus_t {
    check_no_attention(&[dc_attn, dkeys.cast_const()])?;
    let miopen = miopen()?;
    let rnn = as_rnn(rnn_desc)?;
    let (y_desc, _) = rnn_data(y_desc)?;
    let (dx_desc, dx_info) = rnn_data(dx_desc)?;
    let layout = rnn.weight_layout(miopen, dx_info.vector_size, dx_info.data_type)?;
    let workspace = split_workspace(&layout, workspace, workspace_size_in_bytes)?;
    let stream = stream(handle);
    convert_weights(&layout, w, workspace.weights, true, stream)?;
    miopen.miopenRNNBackwardSeqData(
        handle.base,
        rnn.base,
        y_desc.base,
        y,
        dy,
        hx_desc,
        hx,
        dhy,
        dhx,
        cx_desc,
        cx,
        dcy,
        dcx,
        dx_desc.base,
        dx,
        workspace.weights,
        layout.bytes()?,
        workspace.miopen,
        workspace.miopen_size,
        reserve_space,
        reserve_space_size_in_bytes,
    )?;
    zero_padding(&dx_info, dx, stream)
}

pub(crate) unsafe fn rnn_backward_weights_ex(
    handle: &Context,
    rnn_desc: cudnnRNNDescriptor_t,
    x_desc: cudnnRNNDataDescriptor_t,
    x: *const c_void,
    hx_desc: miopenTensorDescriptor_t,
    hx: *const c_void,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *const c_void,
    workspace: *mut c_void,
    workspace_size_in_bytes: usize,
    _dw_desc: miopenTensorDescriptor_t,
    dw: *mut c_void,
    reserve_space: *mut c_void,
    reserve_space_size_in_bytes: usize,
) -> cudnnStatus_t {
    let miopen = miopen()?;
    let rnn = as_rnn(rnn_desc)?;
    let (x_desc, x_info) = rnn_data(x_desc)?;
    let (y_desc, _) = rnn_data(y_desc)?;
    let layout = rnn.weight_layout(miopen, x_info.vector_size, x_info.data_type)?;
    let workspace = split_workspace(&layout, workspace, workspace_size_in_bytes)?;
    let stream = stream(handle);
    let weight_bytes = layout.bytes()?;
    // cuDNN accumulates the gradients into dw. Compute them separately, so it doesn't matter if
    // MIOpen accumulates or overwrites
    hipMemsetAsync(workspace.weights, 0, weight_bytes, stream)
        .map_err(|_| cudnnError_t::EXECUTION_FAILED)?;
    miopen.miopenRNNBackwardWeightsSeqTensor(
        handle.base,
        rnn.base,
        x_desc.base,
        x,
        hx_desc,
        hx,
        y_desc.base,
        y,
        workspace.weights,
        weight_bytes,
        workspace.miopen,
        workspace.miopen_size,
        reserve_space,
        reserve_space_size_in_bytes,
    )?;
    convert_weights(
        &layout,
        workspace.weights,
        workspace.weights_cudnn,
        false,
        stream,
    )?;
    let desc = TensorDescriptor::new(miopen, layout.data_type, &[1, layout.elements as i32])?;
    let (one_f32, zero_f32, one_f64, zero_f64) = (1.0f32, 0.0f32, 1.0f64, 0.0f64);
    let (one, zero): (*const c_void, *const c_void) =
        if layout.data_type == miopenDataType_t::miopenDouble {
            (
                ptr::from_ref(&one_f64).cast(),
                ptr::from_ref(&zero_f64).cast(),
            )
        } else {
            (
                ptr::from_ref(&one_f32).cast(),
                ptr::from_ref(&zero_f32).cast(),
            )
        };
    miopen.miopenOpTensor(
        handle.base,
        miopenTensorOp_t::miopenTensorOpAdd,
        one,
        desc.1,
        dw,
        one,
        desc.1,
        workspace.weights_cudnn,
        zero,
        desc.1,
        dw,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zluda_common::FromCuda;

    #[derive(Clone, Copy)]
    struct Config {
        mode: cudnnRNNMode_t,
        hidden: usize,
        layers: usize,
        directions: usize,
        input: usize,
        seq_length: usize,
        batch: usize,
    }

    impl Config {
        fn gates(&self) -> usize {
            match self.mode {
                cudnnRNNMode_t::CUDNN_LSTM => 4,
                cudnnRNNMode_t::CUDNN_GRU => 3,
                _ => 1,
            }
        }

        fn pseudo_layers(&self) -> usize {
            self.layers * self.directions
        }

        fn layer_input(&self, pseudo_layer: usize) -> usize {
            if pseudo_layer < self.directions {
                self.input
            } else {
                self.hidden * self.directions
            }
        }

        fn matrix_len(&self, pseudo_layer: usize, lin_id: usize) -> usize {
            self.hidden
                * if lin_id < self.gates() {
                    self.layer_input(pseudo_layer)
                } else {
                    self.hidden
                }
        }

        // Offsets in the cuDNN layout: all matrices, then all biases
        fn matrix_offset(&self, pseudo_layer: usize, lin_id: usize) -> usize {
            let mut offset = 0;
            for p in 0..self.pseudo_layers() {
                for l in 0..2 * self.gates() {
                    if (p, l) == (pseudo_layer, lin_id) {
                        return offset;
                    }
                    offset += self.matrix_len(p, l);
                }
            }
            unreachable!()
        }

        fn matrices_len(&self) -> usize {
            (0..self.pseudo_layers())
                .flat_map(|p| (0..2 * self.gates()).map(move |l| (p, l)))
                .map(|(p, l)| self.matrix_len(p, l))
                .sum()
        }

        fn bias_offset(&self, pseudo_layer: usize, lin_id: usize) -> usize {
            self.matrices_len() + (pseudo_layer * 2 * self.gates() + lin_id) * self.hidden
        }

        fn params(&self) -> usize {
            self.matrices_len() + self.pseudo_layers() * 2 * self.gates() * self.hidden
        }

        fn output(&self) -> usize {
            self.hidden * self.directions
        }

        fn state_len(&self) -> usize {
            self.pseudo_layers() * self.batch * self.hidden
        }
    }

    struct Inputs {
        w: Vec<f64>,
        x: Vec<f64>,
        hx: Vec<f64>,
        cx: Vec<f64>,
    }

    struct Outputs {
        y: Vec<f64>,
        hy: Vec<f64>,
        cy: Vec<f64>,
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // Reference implementation of the cuDNN RNN equations
    fn reference_forward(config: &Config, lengths: &[usize], inputs: &Inputs) -> Outputs {
        let (h_size, batch) = (config.hidden, config.batch);
        let w = &inputs.w;
        // W x + bW and R h + bR for one gate
        let affine = |p: usize, lin_id: usize, v: &[f64], r: usize| -> f64 {
            let cols = v.len();
            let m = config.matrix_offset(p, lin_id);
            let b = config.bias_offset(p, lin_id);
            (0..cols).map(|j| w[m + r * cols + j] * v[j]).sum::<f64>() + w[b + r]
        };
        let gates = config.gates();
        let mut layer_input = (0..config.seq_length)
            .map(|t| {
                (0..batch)
                    .map(|b| {
                        let start = (t * batch + b) * config.input;
                        inputs.x[start..start + config.input].to_vec()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut hy = vec![0.0; config.state_len()];
        let mut cy = vec![0.0; config.state_len()];
        for layer in 0..config.layers {
            let mut layer_output = vec![vec![vec![0.0; config.output()]; batch]; config.seq_length];
            for direction in 0..config.directions {
                let p = layer * config.directions + direction;
                for b in 0..batch {
                    let state = (p * batch + b) * h_size;
                    let mut h = inputs.hx[state..state + h_size].to_vec();
                    let mut c = inputs.cx[state..state + h_size].to_vec();
                    let steps: Vec<usize> = if direction == 0 {
                        (0..lengths[b]).collect()
                    } else {
                        (0..lengths[b]).rev().collect()
                    };
                    for t in steps {
                        let x = &layer_input[t][b];
                        let gate =
                            |g: usize, r: usize| affine(p, g, x, r) + affine(p, gates + g, &h, r);
                        let (new_h, new_c): (Vec<f64>, Vec<f64>) = match config.mode {
                            cudnnRNNMode_t::CUDNN_LSTM => (0..h_size)
                                .map(|r| {
                                    let i = sigmoid(gate(0, r));
                                    let f = sigmoid(gate(1, r));
                                    let g = gate(2, r).tanh();
                                    let o = sigmoid(gate(3, r));
                                    let c = f * c[r] + i * g;
                                    (o * c.tanh(), c)
                                })
                                .unzip(),
                            cudnnRNNMode_t::CUDNN_GRU => (0..h_size)
                                .map(|r| {
                                    let reset = sigmoid(gate(0, r));
                                    let update = sigmoid(gate(1, r));
                                    let new = (affine(p, 2, x, r)
                                        + reset * affine(p, gates + 2, &h, r))
                                    .tanh();
                                    ((1.0 - update) * new + update * h[r], 0.0)
                                })
                                .unzip(),
                            cudnnRNNMode_t::CUDNN_RNN_RELU => {
                                (0..h_size).map(|r| (gate(0, r).max(0.0), 0.0)).unzip()
                            }
                            _ => (0..h_size).map(|r| (gate(0, r).tanh(), 0.0)).unzip(),
                        };
                        h = new_h;
                        c = new_c;
                        layer_output[t][b][direction * h_size..(direction + 1) * h_size]
                            .copy_from_slice(&h);
                    }
                    hy[state..state + h_size].copy_from_slice(&h);
                    cy[state..state + h_size].copy_from_slice(&c);
                }
            }
            layer_input = layer_output;
        }
        Outputs {
            y: layer_input.into_iter().flatten().flatten().collect(),
            hy,
            cy,
        }
    }

    struct Random(u64);

    impl Random {
        fn values(&mut self, n: usize) -> Vec<f64> {
            (0..n)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((self.0 >> 40) as f64 / (1u64 << 24) as f64) - 0.5
                })
                .collect()
        }
    }

    struct Handle(cudnnHandle_t);

    impl Drop for Handle {
        fn drop(&mut self) {
            crate::r#impl::destroy(self.0).unwrap();
        }
    }

    unsafe fn tensor(dims: &[i32]) -> miopenTensorDescriptor_t {
        let mut desc = mem::zeroed();
        crate::r#impl::create_tensor_descriptor(&mut desc).unwrap();
        let mut strides = vec![1i32; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        crate::r#impl::set_tensor_nd_descriptor(
            desc,
            miopenDataType_t::miopenFloat,
            dims.len() as i32,
            dims.as_ptr(),
            strides.as_ptr(),
        )
        .unwrap();
        desc
    }

    unsafe fn alloc(size: usize) -> *mut c_void {
        let mut ptr = ptr::null_mut();
        hipMalloc(&mut ptr, size.max(1)).unwrap();
        ptr
    }

    unsafe fn upload(data: &[f64]) -> *mut c_void {
        let data = data.iter().map(|x| *x as f32).collect::<Vec<_>>();
        let ptr = alloc(data.len() * 4);
        hipMemcpy(
            ptr,
            data.as_ptr().cast(),
            data.len() * 4,
            hipMemcpyKind::hipMemcpyHostToDevice,
        )
        .unwrap();
        ptr
    }

    unsafe fn download(ptr: *mut c_void, len: usize) -> Vec<f64> {
        let mut data = vec![0f32; len];
        hipMemcpy(
            data.as_mut_ptr().cast(),
            ptr,
            len * 4,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        )
        .unwrap();
        hipFree(ptr).unwrap();
        data.into_iter().map(|x| x as f64).collect()
    }

    struct Gradients {
        dy: Vec<f64>,
        dhy: Vec<f64>,
        dcy: Vec<f64>,
    }

    struct GpuResults {
        outputs: Outputs,
        dx: Vec<f64>,
        dhx: Vec<f64>,
        dcx: Vec<f64>,
        dw: Vec<f64>,
    }

    // Runs forward training, backward data and backward weights like MXNet does
    unsafe fn run_gpu(
        config: &Config,
        lengths: &[usize],
        padded: bool,
        with_dropout: bool,
        inputs: &Inputs,
        gradients: &Gradients,
        dw_init: &[f64],
    ) -> GpuResults {
        let mut handle = ptr::null_mut();
        crate::r#impl::create(&mut handle).unwrap();
        let handle = Handle(handle);
        let context: &Context = FromCuda::<_, cudnnError_t>::from_cuda(&handle.0).unwrap();
        let mut dropout = miopenDropoutDescriptor_t(ptr::null_mut());
        if with_dropout {
            // Like MXNet: a dropout of 0 without random number generator states
            crate::r#impl::create_dropout_descriptor(&mut dropout).unwrap();
            crate::r#impl::set_dropout_descriptor(dropout, context, 0.0, ptr::null_mut(), 0, 0)
                .unwrap();
        }
        let mut rnn = ptr::null_mut();
        create_rnn_descriptor(&mut rnn).unwrap();
        set_rnn_descriptor_v6(
            context,
            rnn,
            config.hidden as i32,
            config.layers as i32,
            dropout,
            cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            if config.directions == 2 {
                cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL
            } else {
                cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL
            },
            config.mode,
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
            miopenDataType_t::miopenFloat,
        )
        .unwrap();
        let x_step = tensor(&[config.batch as i32, config.input as i32, 1]);
        let mut params_size = 0;
        get_rnn_params_size(
            context,
            rnn,
            x_step,
            &mut params_size,
            miopenDataType_t::miopenFloat,
        )
        .unwrap();
        assert_eq!(params_size, config.params() * 4);
        let (mut workspace_size, mut reserve_size) = (0, 0);
        get_rnn_workspace_size(
            context,
            rnn,
            config.seq_length as i32,
            &x_step,
            &mut workspace_size,
        )
        .unwrap();
        get_rnn_training_reserve_size(
            context,
            rnn,
            config.seq_length as i32,
            &x_step,
            &mut reserve_size,
        )
        .unwrap();
        let lengths_i32 = lengths.iter().map(|x| *x as i32).collect::<Vec<_>>();
        let layout = if padded {
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
        } else {
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
        };
        let mut padding_fill = 0f32;
        let mut data_desc = |vector_size: usize| {
            let mut desc = ptr::null_mut();
            create_rnn_data_descriptor(&mut desc).unwrap();
            set_rnn_data_descriptor(
                desc,
                miopenDataType_t::miopenFloat,
                layout,
                config.seq_length as i32,
                config.batch as i32,
                vector_size as i32,
                lengths_i32.as_ptr(),
                ptr::from_mut(&mut padding_fill).cast(),
            )
            .unwrap();
            desc
        };
        let x_desc = data_desc(config.input);
        let y_desc = data_desc(config.output());
        let h_desc = tensor(&[
            config.pseudo_layers() as i32,
            config.batch as i32,
            config.hidden as i32,
        ]);
        let is_lstm = config.mode == cudnnRNNMode_t::CUDNN_LSTM;
        let y_len = config.seq_length * config.batch * config.output();
        let x_len = config.seq_length * config.batch * config.input;
        let (w, x, hx) = (upload(&inputs.w), upload(&inputs.x), upload(&inputs.hx));
        let cx = if is_lstm {
            upload(&inputs.cx)
        } else {
            ptr::null_mut()
        };
        let y = alloc(y_len * 4);
        let hy = alloc(config.state_len() * 4);
        let cy = if is_lstm {
            alloc(config.state_len() * 4)
        } else {
            ptr::null_mut()
        };
        let workspace = alloc(workspace_size);
        let reserve = alloc(reserve_size);
        // Attention descriptors and buffers are unused
        let (no_desc, null): (cudnnRNNDataDescriptor_t, *mut c_void) =
            (ptr::null_mut(), ptr::null_mut());
        rnn_forward_training_ex(
            context,
            rnn,
            x_desc,
            x,
            h_desc,
            hx,
            h_desc,
            cx,
            h_desc,
            w,
            y_desc,
            y,
            h_desc,
            hy,
            h_desc,
            cy,
            no_desc,
            null,
            no_desc,
            null,
            no_desc,
            null,
            no_desc,
            null,
            workspace,
            workspace_size,
            reserve,
            reserve_size,
        )
        .unwrap();
        let dy = upload(&gradients.dy);
        let dhy = upload(&gradients.dhy);
        let dcy = if is_lstm {
            upload(&gradients.dcy)
        } else {
            ptr::null_mut()
        };
        let dx = alloc(x_len * 4);
        let dhx = alloc(config.state_len() * 4);
        let dcx = if is_lstm {
            alloc(config.state_len() * 4)
        } else {
            ptr::null_mut()
        };
        rnn_backward_data_ex(
            context,
            rnn,
            y_desc,
            y,
            y_desc,
            dy,
            no_desc,
            null,
            h_desc,
            dhy,
            h_desc,
            dcy,
            h_desc,
            w,
            h_desc,
            hx,
            h_desc,
            cx,
            x_desc,
            dx,
            h_desc,
            dhx,
            h_desc,
            dcx,
            no_desc,
            null,
            workspace,
            workspace_size,
            reserve,
            reserve_size,
        )
        .unwrap();
        let dw = upload(dw_init);
        rnn_backward_weights_ex(
            context,
            rnn,
            x_desc,
            x,
            h_desc,
            hx,
            y_desc,
            y,
            workspace,
            workspace_size,
            h_desc,
            dw,
            reserve,
            reserve_size,
        )
        .unwrap();
        hipDeviceSynchronize().unwrap();
        let results = GpuResults {
            outputs: Outputs {
                y: download(y, y_len),
                hy: download(hy, config.state_len()),
                cy: if is_lstm {
                    download(cy, config.state_len())
                } else {
                    Vec::new()
                },
            },
            dx: download(dx, x_len),
            dhx: download(dhx, config.state_len()),
            dcx: if is_lstm {
                download(dcx, config.state_len())
            } else {
                Vec::new()
            },
            dw: download(dw, config.params()),
        };
        for ptr in [w, x, hx, cx, dy, dhy, dcy, workspace, reserve] {
            if !ptr.is_null() {
                hipFree(ptr).unwrap();
            }
        }
        for desc in [x_desc, y_desc] {
            destroy_rnn_data_descriptor(desc).unwrap();
        }
        for desc in [x_step, h_desc] {
            crate::r#impl::destroy_tensor_descriptor(desc).unwrap();
        }
        destroy_rnn_descriptor(rnn).unwrap();
        if with_dropout {
            crate::r#impl::destroy_dropout_descriptor(dropout).unwrap();
        }
        results
    }

    fn loss(config: &Config, lengths: &[usize], inputs: &Inputs, gradients: &Gradients) -> f64 {
        let outputs = reference_forward(config, lengths, inputs);
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        dot(&outputs.y, &gradients.dy)
            + dot(&outputs.hy, &gradients.dhy)
            + if config.mode == cudnnRNNMode_t::CUDNN_LSTM {
                dot(&outputs.cy, &gradients.dcy)
            } else {
                0.0
            }
    }

    // Gradient of the loss with respect to one of the inputs, by central differences
    fn numerical_gradient(
        config: &Config,
        lengths: &[usize],
        inputs: &mut Inputs,
        gradients: &Gradients,
        select: fn(&mut Inputs) -> &mut Vec<f64>,
    ) -> Vec<f64> {
        let eps = 1e-6;
        (0..select(inputs).len())
            .map(|i| {
                let original = select(inputs)[i];
                select(inputs)[i] = original + eps;
                let plus = loss(config, lengths, inputs, gradients);
                select(inputs)[i] = original - eps;
                let minus = loss(config, lengths, inputs, gradients);
                select(inputs)[i] = original;
                (plus - minus) / (2.0 * eps)
            })
            .collect()
    }

    #[track_caller]
    fn assert_close(name: &str, actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-3 + 1e-3 * e.abs(),
                "{name}[{i}]: {a} != {e}\nactual: {actual:?}\nexpected: {expected:?}"
            );
        }
    }

    fn check(config: Config, lengths: &[usize], padded: bool, with_dropout: bool) {
        let mut random = Random(config.params() as u64 * 7919 + config.mode.0 as u64);
        let mut inputs = Inputs {
            w: random.values(config.params()),
            x: random.values(config.seq_length * config.batch * config.input),
            hx: random.values(config.state_len()),
            cx: random.values(config.state_len()),
        };
        if config.mode != cudnnRNNMode_t::CUDNN_LSTM {
            inputs.cx = vec![0.0; config.state_len()];
        }
        // Inputs and output gradients in the padding are ignored
        let y_len = config.seq_length * config.batch * config.output();
        let gradients = Gradients {
            dy: random.values(y_len),
            dhy: random.values(config.state_len()),
            dcy: random.values(config.state_len()),
        };
        let dw_init = random.values(config.params());
        let gpu = unsafe {
            run_gpu(
                &config,
                lengths,
                padded,
                with_dropout,
                &inputs,
                &gradients,
                &dw_init,
            )
        };
        let expected = reference_forward(&config, lengths, &inputs);
        assert_close("y", &gpu.outputs.y, &expected.y);
        assert_close("hy", &gpu.outputs.hy, &expected.hy);
        let mut dx = numerical_gradient(&config, lengths, &mut inputs, &gradients, |i| &mut i.x);
        // dx in the padding is zero
        for t in 0..config.seq_length {
            for b in 0..config.batch {
                if t >= lengths[b] {
                    let start = (t * config.batch + b) * config.input;
                    dx[start..start + config.input].fill(0.0);
                }
            }
        }
        assert_close("dx", &gpu.dx, &dx);
        let dhx = numerical_gradient(&config, lengths, &mut inputs, &gradients, |i| &mut i.hx);
        assert_close("dhx", &gpu.dhx, &dhx);
        if config.mode == cudnnRNNMode_t::CUDNN_LSTM {
            assert_close("cy", &gpu.outputs.cy, &expected.cy);
            let dcx = numerical_gradient(&config, lengths, &mut inputs, &gradients, |i| &mut i.cx);
            assert_close("dcx", &gpu.dcx, &dcx);
        }
        // dw is accumulated
        let dw = numerical_gradient(&config, lengths, &mut inputs, &gradients, |i| &mut i.w)
            .iter()
            .zip(&dw_init)
            .map(|(g, init)| g + init)
            .collect::<Vec<_>>();
        assert_close("dw", &gpu.dw, &dw);
    }

    fn config(mode: cudnnRNNMode_t, layers: usize, directions: usize) -> Config {
        Config {
            mode,
            hidden: 3,
            layers,
            directions,
            input: 2,
            seq_length: 4,
            batch: 2,
        }
    }

    #[test]
    fn lstm() {
        check(
            config(cudnnRNNMode_t::CUDNN_LSTM, 1, 1),
            &[4, 4],
            false,
            false,
        );
    }

    #[test]
    fn lstm_stacked_bidirectional() {
        check(
            config(cudnnRNNMode_t::CUDNN_LSTM, 2, 2),
            &[4, 4],
            false,
            true,
        );
    }

    #[test]
    fn lstm_padded() {
        check(
            config(cudnnRNNMode_t::CUDNN_LSTM, 1, 2),
            &[2, 4],
            true,
            true,
        );
    }

    #[test]
    fn gru() {
        check(
            config(cudnnRNNMode_t::CUDNN_GRU, 1, 1),
            &[4, 4],
            false,
            false,
        );
    }

    #[test]
    fn gru_stacked_bidirectional_padded() {
        check(config(cudnnRNNMode_t::CUDNN_GRU, 2, 2), &[3, 1], true, true);
    }

    #[test]
    fn rnn_tanh() {
        check(
            config(cudnnRNNMode_t::CUDNN_RNN_TANH, 1, 1),
            &[4, 4],
            false,
            false,
        );
    }

    #[test]
    fn rnn_relu_stacked() {
        check(
            config(cudnnRNNMode_t::CUDNN_RNN_RELU, 2, 1),
            &[4, 4],
            false,
            true,
        );
    }
}
