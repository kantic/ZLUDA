// Implementation of the cuDNN backend (graph) API on top of MIOpen.
//
// Only single-operation graphs with a convolution (forward, backward data or backward filter)
// are supported. This is what frameworks like MXNet use for their convolution layers.
//
// Every graph is served by a single engine (global index 0) without knobs. The engine runs
// the fastest MIOpen solution for the convolution problem.

use super::{miopen, run_solution, Context};
use cuda_types::cudnn9::*;
use miopen_sys::*;
use rustc_hash::FxHashMap;
use std::{ffi::c_void, mem, ptr, sync::Arc, sync::Mutex};
use zluda_common::FromCuda;

const DESCRIPTOR_COOKIE: usize = 0x9c3b_5d1e_27a4_f601;

pub(crate) struct Descriptor {
    cookie: usize,
    state: Mutex<State>,
}

struct State {
    type_: cudnnBackendDescriptorType_t,
    attributes: Attributes,
    finalized: Option<Arc<Content>>,
}

type Attributes = FxHashMap<u32, Attribute>;

#[derive(Clone)]
enum Attribute {
    Values {
        type_: cudnnBackendAttributeType_t,
        count: usize,
        bytes: Vec<u8>,
    },
    Descriptors(Vec<Arc<Content>>),
}

// Immutable content of a finalized descriptor. Descriptors referenced by other descriptors are
// captured by value, so they can be destroyed independently.
struct Content {
    type_: cudnnBackendDescriptorType_t,
    attributes: Attributes,
    plan: Option<ConvolutionPlan>,
}

unsafe impl Send for Content {}
unsafe impl Sync for Content {}

fn element_size(type_: cudnnBackendAttributeType_t) -> usize {
    match type_ {
        cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE
        | cudnnBackendAttributeType_t::CUDNN_TYPE_INT64
        | cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE
        | cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR
        | cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR => 8,
        cudnnBackendAttributeType_t::CUDNN_TYPE_BOOLEAN
        | cudnnBackendAttributeType_t::CUDNN_TYPE_CHAR => 1,
        cudnnBackendAttributeType_t::CUDNN_TYPE_FRACTION => 16,
        // FLOAT, INT32 and all the enums
        _ => 4,
    }
}

unsafe fn as_descriptor<'a>(
    desc: cudnnBackendDescriptor_t,
) -> Result<&'a Descriptor, cudnnError_t> {
    let desc = desc
        .cast::<Descriptor>()
        .as_ref()
        .ok_or(cudnnError_t::BAD_PARAM)?;
    if desc.cookie != DESCRIPTOR_COOKIE {
        return Err(cudnnError_t::BAD_PARAM);
    }
    Ok(desc)
}

fn lock(desc: &Descriptor) -> Result<std::sync::MutexGuard<'_, State>, cudnnError_t> {
    desc.state.lock().map_err(|_| cudnnError_t::INTERNAL_ERROR)
}

pub(crate) unsafe fn backend_create_descriptor(
    descriptor_type: cudnnBackendDescriptorType_t,
    descriptor: &mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t {
    *descriptor = Box::into_raw(Box::new(Descriptor {
        cookie: DESCRIPTOR_COOKIE,
        state: Mutex::new(State {
            type_: descriptor_type,
            attributes: FxHashMap::default(),
            finalized: None,
        }),
    }))
    .cast();
    Ok(())
}

pub(crate) unsafe fn backend_destroy_descriptor(
    descriptor: cudnnBackendDescriptor_t,
) -> cudnnStatus_t {
    as_descriptor(descriptor)?;
    let mut desc = Box::from_raw(descriptor.cast::<Descriptor>());
    desc.cookie = 0;
    drop(desc);
    Ok(())
}

pub(crate) unsafe fn backend_initialize(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t {
    let desc = as_descriptor(descriptor)?;
    let mut state = lock(desc)?;
    state.attributes.clear();
    state.finalized = None;
    Ok(())
}

pub(crate) unsafe fn backend_set_attribute(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    element_count: i64,
    array_of_elements: *const c_void,
) -> cudnnStatus_t {
    let desc = as_descriptor(descriptor)?;
    if element_count < 0 || (element_count > 0 && array_of_elements.is_null()) {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let count = element_count as usize;
    let attribute = if attribute_type == cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR
    {
        let elements = if count == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(array_of_elements.cast::<cudnnBackendDescriptor_t>(), count)
        };
        let mut contents = Vec::with_capacity(count);
        for element in elements {
            let element = as_descriptor(*element)?;
            let element_state = lock(element)?;
            contents.push(
                element_state
                    .finalized
                    .clone()
                    .ok_or(cudnnError_t::BAD_PARAM)?,
            );
        }
        Attribute::Descriptors(contents)
    } else {
        let size = count * element_size(attribute_type);
        let bytes = if size == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(array_of_elements.cast::<u8>(), size).to_vec()
        };
        Attribute::Values {
            type_: attribute_type,
            count,
            bytes,
        }
    };
    let mut state = lock(desc)?;
    if state.finalized.is_some() {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    state.attributes.insert(attribute_name.0, attribute);
    Ok(())
}

pub(crate) unsafe fn backend_get_attribute(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    requested_element_count: i64,
    element_count: *mut i64,
    array_of_elements: *mut c_void,
) -> cudnnStatus_t {
    let desc = as_descriptor(descriptor)?;
    if requested_element_count < 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    // Clone the attribute, so no lock is held while the target descriptors are written
    let attribute = {
        let state = lock(desc)?;
        let attributes = match state.finalized {
            Some(ref content) => &content.attributes,
            None => &state.attributes,
        };
        attributes
            .get(&attribute_name.0)
            .cloned()
            .ok_or(cudnnError_t::BAD_PARAM)?
    };
    let requested = requested_element_count as usize;
    match attribute {
        Attribute::Values {
            type_,
            count,
            bytes,
        } => {
            if type_ != attribute_type {
                return Err(cudnnError_t::BAD_PARAM);
            }
            if let Some(element_count) = element_count.as_mut() {
                *element_count = count as i64;
            }
            let copied = requested.min(count);
            if copied > 0 {
                if array_of_elements.is_null() {
                    return Err(cudnnError_t::BAD_PARAM);
                }
                ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    array_of_elements.cast::<u8>(),
                    copied * element_size(type_),
                );
            }
        }
        Attribute::Descriptors(contents) => {
            if attribute_type != cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR {
                return Err(cudnnError_t::BAD_PARAM);
            }
            if let Some(element_count) = element_count.as_mut() {
                *element_count = contents.len() as i64;
            }
            let copied = requested.min(contents.len());
            if copied > 0 {
                if array_of_elements.is_null() {
                    return Err(cudnnError_t::BAD_PARAM);
                }
                let targets = std::slice::from_raw_parts(
                    array_of_elements.cast::<cudnnBackendDescriptor_t>(),
                    copied,
                );
                for (target, content) in targets.iter().zip(contents) {
                    let target = as_descriptor(*target)?;
                    let mut target_state = lock(target)?;
                    if target_state.type_ != content.type_ {
                        return Err(cudnnError_t::BAD_PARAM);
                    }
                    target_state.attributes = content.attributes.clone();
                    target_state.finalized = Some(content);
                }
            }
        }
    }
    Ok(())
}

pub(crate) unsafe fn backend_finalize(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t {
    let desc = as_descriptor(descriptor)?;
    let mut state = lock(desc)?;
    if state.finalized.is_some() {
        return Ok(());
    }
    let content = finalize(state.type_, state.attributes.clone())?;
    state.finalized = Some(Arc::new(content));
    Ok(())
}

pub(crate) unsafe fn backend_execute(
    handle: &Context,
    execution_plan: cudnnBackendDescriptor_t,
    variant_pack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t {
    let plan = finalized_content(execution_plan)?;
    let plan = plan.plan.as_ref().ok_or(cudnnError_t::BAD_PARAM)?;
    let variant_pack = finalized_content(variant_pack)?;
    if variant_pack.type_ != cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let attributes = &variant_pack.attributes;
    let uids = values::<i64>(
        attributes,
        cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
        cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
    )?;
    let pointers = values::<*mut c_void>(
        attributes,
        cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
        cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
    )?;
    let workspace = optional_value::<*mut c_void>(
        attributes,
        cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
        cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
    )?
    .unwrap_or(ptr::null_mut());
    let pointer = |uid: i64| -> Result<*mut c_void, cudnnError_t> {
        uids.iter()
            .position(|x| *x == uid)
            .map(|i| pointers[i])
            .ok_or(cudnnError_t::BAD_PARAM)
    };
    let problem = &plan.problem;
    let x = pointer(problem.x.uid)?;
    let w = pointer(problem.w.uid)?;
    let y = pointer(problem.y.uid)?;
    let (target_desc, target) = match problem.direction {
        miopenProblemDirection_t::miopenProblemDirectionForward => (plan.y, y),
        miopenProblemDirection_t::miopenProblemDirectionBackward => (plan.x, x),
        _ => (plan.w, w),
    };
    let is_f64 = problem.x.data_type == cudnnDataType_t::CUDNN_DATA_DOUBLE;
    let (alpha_f32, beta_f32) = (problem.alpha as f32, problem.beta as f32);
    let (alpha, beta): (*const c_void, *const c_void) = if is_f64 {
        (
            ptr::from_ref(&problem.alpha).cast(),
            ptr::from_ref(&problem.beta).cast(),
        )
    } else {
        (
            ptr::from_ref(&alpha_f32).cast(),
            ptr::from_ref(&beta_f32).cast(),
        )
    };
    run_solution(
        handle,
        problem.direction,
        alpha,
        beta,
        plan.x,
        x,
        plan.w,
        w,
        plan.y,
        y,
        target_desc,
        target,
        plan.conv,
        plan.algo,
        workspace,
        plan.workspace_size,
    )?;
    Ok(())
}

unsafe fn finalized_content(desc: cudnnBackendDescriptor_t) -> Result<Arc<Content>, cudnnError_t> {
    let desc = as_descriptor(desc)?;
    let state = lock(desc)?;
    state.finalized.clone().ok_or(cudnnError_t::BAD_PARAM)
}

fn attribute<'a>(
    attributes: &'a Attributes,
    name: cudnnBackendAttributeName_t,
) -> Result<&'a Attribute, cudnnError_t> {
    attributes.get(&name.0).ok_or(cudnnError_t::BAD_PARAM)
}

// Read all elements of a value attribute. T must match the attribute type
fn values<T: Copy>(
    attributes: &Attributes,
    name: cudnnBackendAttributeName_t,
    type_: cudnnBackendAttributeType_t,
) -> Result<Vec<T>, cudnnError_t> {
    match attribute(attributes, name)? {
        Attribute::Values {
            type_: actual_type,
            count,
            bytes,
        } if *actual_type == type_ && element_size(type_) == mem::size_of::<T>() => Ok((0..*count)
            .map(|i| unsafe { bytes.as_ptr().cast::<T>().add(i).read_unaligned() })
            .collect()),
        _ => Err(cudnnError_t::BAD_PARAM),
    }
}

fn value<T: Copy>(
    attributes: &Attributes,
    name: cudnnBackendAttributeName_t,
    type_: cudnnBackendAttributeType_t,
) -> Result<T, cudnnError_t> {
    values::<T>(attributes, name, type_)?
        .first()
        .copied()
        .ok_or(cudnnError_t::BAD_PARAM)
}

fn optional_value<T: Copy>(
    attributes: &Attributes,
    name: cudnnBackendAttributeName_t,
    type_: cudnnBackendAttributeType_t,
) -> Result<Option<T>, cudnnError_t> {
    if attributes.contains_key(&name.0) {
        Ok(Some(value(attributes, name, type_)?))
    } else {
        Ok(None)
    }
}

// Read a scaling factor, which can be set as either float or double
fn optional_scale(
    attributes: &Attributes,
    name: cudnnBackendAttributeName_t,
) -> Result<Option<f64>, cudnnError_t> {
    match attributes.get(&name.0) {
        None => Ok(None),
        Some(Attribute::Values {
            type_: cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
            ..
        }) => Ok(Some(value::<f64>(
            attributes,
            name,
            cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
        )?)),
        Some(_) => Ok(Some(value::<f32>(
            attributes,
            name,
            cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
        )? as f64)),
    }
}

fn descriptors<'a>(
    attributes: &'a Attributes,
    name: cudnnBackendAttributeName_t,
    type_: cudnnBackendDescriptorType_t,
) -> Result<&'a [Arc<Content>], cudnnError_t> {
    match attribute(attributes, name)? {
        Attribute::Descriptors(contents) if contents.iter().all(|c| c.type_ == type_) => {
            Ok(contents)
        }
        _ => Err(cudnnError_t::BAD_PARAM),
    }
}

fn descriptor<'a>(
    attributes: &'a Attributes,
    name: cudnnBackendAttributeName_t,
    type_: cudnnBackendDescriptorType_t,
) -> Result<&'a Arc<Content>, cudnnError_t> {
    descriptors(attributes, name, type_)?
        .first()
        .ok_or(cudnnError_t::BAD_PARAM)
}

fn int64_attribute(value: i64) -> Attribute {
    Attribute::Values {
        type_: cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        count: 1,
        bytes: value.to_ne_bytes().to_vec(),
    }
}

fn empty_attribute(type_: cudnnBackendAttributeType_t) -> Attribute {
    Attribute::Values {
        type_,
        count: 0,
        bytes: Vec::new(),
    }
}

fn finalize(
    type_: cudnnBackendDescriptorType_t,
    mut attributes: Attributes,
) -> Result<Content, cudnnError_t> {
    let mut plan = None;
    match type_ {
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR => {
            TensorParams::new(&attributes)?;
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR => {
            ConvolutionParams::new(&attributes)?;
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
        | cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
        | cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR => {
            ConvolutionProblem::from_operation(type_, &attributes)?;
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR => {
            value::<cudnnHandle_t>(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE,
            )?;
            match attribute(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS,
            )? {
                Attribute::Descriptors(ops) if !ops.is_empty() => {}
                _ => return Err(cudnnError_t::BAD_PARAM),
            }
            // We have a single engine
            attributes.insert(
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT.0,
                int64_attribute(1),
            );
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR => {
            let graph = descriptor(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
            )?
            .clone();
            let results = if ConvolutionProblem::from_graph(&graph).is_ok() {
                let mut engine_attributes = Attributes::default();
                engine_attributes.insert(
                    cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH.0,
                    Attribute::Descriptors(vec![graph]),
                );
                engine_attributes.insert(
                    cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_GLOBAL_INDEX.0,
                    int64_attribute(0),
                );
                let engine = finalize(
                    cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR,
                    engine_attributes,
                )?;
                let mut config_attributes = Attributes::default();
                config_attributes.insert(
                    cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE.0,
                    Attribute::Descriptors(vec![Arc::new(engine)]),
                );
                let config = finalize(
                    cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                    config_attributes,
                )?;
                vec![Arc::new(config)]
            } else {
                Vec::new()
            };
            attributes.insert(
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_RESULTS.0,
                Attribute::Descriptors(results),
            );
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR => {
            let index = value::<i64>(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            )?;
            let graph = descriptor(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
            )?;
            if index != 0 || ConvolutionProblem::from_graph(graph).is_err() {
                return Err(cudnnError_t::NOT_SUPPORTED);
            }
            attributes.insert(
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_NUMERICAL_NOTE.0,
                empty_attribute(cudnnBackendAttributeType_t::CUDNN_TYPE_NUMERICAL_NOTE),
            );
            attributes.insert(
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE.0,
                empty_attribute(cudnnBackendAttributeType_t::CUDNN_TYPE_BEHAVIOR_NOTE),
            );
            attributes.insert(
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_KNOB_INFO.0,
                Attribute::Descriptors(Vec::new()),
            );
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR => {
            descriptor(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE,
                cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR,
            )?;
            // The engine has no knobs, but the choices can still be set (to an empty list)
            attributes
                .entry(cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_KNOB_CHOICES.0)
                .or_insert(Attribute::Descriptors(Vec::new()));
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR => {
            let handle = value::<cudnnHandle_t>(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE,
            )?;
            let handle: &Context = FromCuda::<_, cudnnError_t>::from_cuda(&handle)?;
            let config = descriptor(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
            )?;
            let engine = descriptor(
                &config.attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE,
                cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR,
            )?;
            let graph = descriptor(
                &engine.attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
            )?;
            let problem = ConvolutionProblem::from_graph(graph)?;
            let new_plan = unsafe { ConvolutionPlan::new(handle, problem) }?;
            attributes.insert(
                cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE.0,
                int64_attribute(new_plan.workspace_size as i64),
            );
            plan = Some(new_plan);
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR => {
            let uids = values::<i64>(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            )?;
            let pointers = values::<*mut c_void>(
                &attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
            )?;
            if uids.len() != pointers.len() {
                return Err(cudnnError_t::BAD_PARAM);
            }
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR => {}
        _ => return Err(cudnnError_t::NOT_SUPPORTED),
    }
    Ok(Content {
        type_,
        attributes,
        plan,
    })
}

struct TensorParams {
    uid: i64,
    data_type: cudnnDataType_t,
    dimensions: Vec<i64>,
    strides: Vec<i64>,
}

impl TensorParams {
    fn new(attributes: &Attributes) -> Result<Self, cudnnError_t> {
        let uid = value::<i64>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )?;
        let data_type = value::<cudnnDataType_t>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DATA_TYPE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
        )?;
        let dimensions = values::<i64>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DIMENSIONS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )?;
        let strides = values::<i64>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )?;
        if dimensions.is_empty() || dimensions.len() != strides.len() || dimensions.len() > 8 {
            return Err(cudnnError_t::BAD_PARAM);
        }
        Ok(Self {
            uid,
            data_type,
            dimensions,
            strides,
        })
    }

    fn from_content(content: &Content) -> Result<Self, cudnnError_t> {
        if content.type_ != cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR {
            return Err(cudnnError_t::BAD_PARAM);
        }
        Self::new(&content.attributes)
    }
}

struct ConvolutionParams {
    mode: cudnnConvolutionMode_t,
    strides: Vec<i64>,
    dilations: Vec<i64>,
    paddings: Vec<i64>,
    post_paddings: Vec<i64>,
}

impl ConvolutionParams {
    fn new(attributes: &Attributes) -> Result<Self, cudnnError_t> {
        let spatial_dims = value::<i64>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )?;
        let mode = value::<cudnnConvolutionMode_t>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_CONV_MODE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_CONVOLUTION_MODE,
        )?;
        let strides = values::<i64>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )?;
        let dilations = values::<i64>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_DILATIONS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )?;
        let paddings = values::<i64>(
            attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )?;
        let post_paddings = if attributes
            .contains_key(&cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_POST_PADDINGS.0)
        {
            values::<i64>(
                attributes,
                cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            )?
        } else {
            paddings.clone()
        };
        let spatial_dims = spatial_dims as usize;
        if spatial_dims == 0
            || strides.len() != spatial_dims
            || dilations.len() != spatial_dims
            || paddings.len() != spatial_dims
            || post_paddings.len() != spatial_dims
        {
            return Err(cudnnError_t::BAD_PARAM);
        }
        Ok(Self {
            mode,
            strides,
            dilations,
            paddings,
            post_paddings,
        })
    }

    // Checks if the convolution can be executed by MIOpen
    fn check_supported(&self) -> Result<(), cudnnError_t> {
        if self.mode != cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION
            || self.post_paddings != self.paddings
        {
            return Err(cudnnError_t::NOT_SUPPORTED);
        }
        Ok(())
    }
}

// A convolution in the MIOpen convention: x is the input, w the filter, y the output,
// regardless of the direction
struct ConvolutionProblem {
    direction: miopenProblemDirection_t,
    convolution: ConvolutionParams,
    x: TensorParams,
    w: TensorParams,
    y: TensorParams,
    alpha: f64,
    beta: f64,
}

impl ConvolutionProblem {
    fn from_graph(graph: &Content) -> Result<Self, cudnnError_t> {
        let ops = match attribute(
            &graph.attributes,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS,
        )? {
            Attribute::Descriptors(ops) => ops,
            _ => return Err(cudnnError_t::BAD_PARAM),
        };
        match &ops[..] {
            [op] => Self::from_operation(op.type_, &op.attributes),
            _ => Err(cudnnError_t::NOT_SUPPORTED),
        }
    }

    fn from_operation(
        type_: cudnnBackendDescriptorType_t,
        attributes: &Attributes,
    ) -> Result<Self, cudnnError_t> {
        let (direction, conv, x, w, y, alpha, beta) = match type_ {
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR => (
                miopenProblemDirection_t::miopenProblemDirectionForward,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
            ),
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR => (
                miopenProblemDirection_t::miopenProblemDirectionBackward,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
            ),
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR => (
                miopenProblemDirection_t::miopenProblemDirectionBackwardWeights,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
            ),
            _ => return Err(cudnnError_t::NOT_SUPPORTED),
        };
        let tensor = |name| -> Result<TensorParams, cudnnError_t> {
            TensorParams::from_content(descriptor(
                attributes,
                name,
                cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR,
            )?)
        };
        let conv = descriptor(
            attributes,
            conv,
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
        )?;
        let convolution = ConvolutionParams::new(&conv.attributes)?;
        convolution.check_supported()?;
        let problem = Self {
            direction,
            convolution,
            x: tensor(x)?,
            w: tensor(w)?,
            y: tensor(y)?,
            alpha: optional_scale(attributes, alpha)?.unwrap_or(1.0),
            beta: optional_scale(attributes, beta)?.unwrap_or(0.0),
        };
        let spatial_dims = problem.convolution.strides.len();
        if [&problem.x, &problem.w, &problem.y]
            .iter()
            .any(|t| t.dimensions.len() != spatial_dims + 2 || t.dimensions.len() > 5)
        {
            return Err(cudnnError_t::NOT_SUPPORTED);
        }
        if problem.alpha != 1.0 {
            return Err(cudnnError_t::NOT_SUPPORTED);
        }
        Ok(problem)
    }
}

struct ConvolutionPlan {
    problem: ConvolutionProblem,
    x: miopenTensorDescriptor_t,
    w: miopenTensorDescriptor_t,
    y: miopenTensorDescriptor_t,
    conv: miopenConvolutionDescriptor_t,
    algo: u32,
    workspace_size: usize,
}

fn to_i32(values: &[i64]) -> Result<Vec<i32>, cudnnError_t> {
    values
        .iter()
        .map(|x| i32::try_from(*x).map_err(|_| cudnnError_t::NOT_SUPPORTED))
        .collect()
}

unsafe fn create_tensor(
    miopen: &super::MIOpenVtable,
    tensor: &TensorParams,
) -> Result<miopenTensorDescriptor_t, cudnnError_t> {
    let data_type: miopenDataType_t = FromCuda::<_, cudnnError_t>::from_cuda(&tensor.data_type)?;
    let dimensions = to_i32(&tensor.dimensions)?;
    let strides = to_i32(&tensor.strides)?;
    let mut desc = mem::zeroed();
    miopen.miopenCreateTensorDescriptor(&mut desc)?;
    if let Err(err) = miopen.miopenSetTensorDescriptor(
        desc,
        data_type,
        dimensions.len() as i32,
        dimensions.as_ptr(),
        strides.as_ptr(),
    ) {
        miopen.miopenDestroyTensorDescriptor(desc).ok();
        return Err(err.into());
    }
    Ok(desc)
}

impl ConvolutionPlan {
    unsafe fn new(handle: &Context, problem: ConvolutionProblem) -> Result<Self, cudnnError_t> {
        let miopen = miopen()?;
        let mut plan = Self {
            problem,
            x: miopenTensorDescriptor_t(ptr::null_mut()),
            w: miopenTensorDescriptor_t(ptr::null_mut()),
            y: miopenTensorDescriptor_t(ptr::null_mut()),
            conv: miopenConvolutionDescriptor_t(ptr::null_mut()),
            algo: 0,
            workspace_size: 0,
        };
        // From here on, Drop cleans up the descriptors created so far
        plan.x = create_tensor(miopen, &plan.problem.x)?;
        plan.w = create_tensor(miopen, &plan.problem.w)?;
        plan.y = create_tensor(miopen, &plan.problem.y)?;
        let conv = &plan.problem.convolution;
        let paddings = to_i32(&conv.paddings)?;
        let strides = to_i32(&conv.strides)?;
        let dilations = to_i32(&conv.dilations)?;
        miopen.miopenCreateConvolutionDescriptor(&mut plan.conv)?;
        miopen.miopenInitConvolutionNdDescriptor(
            plan.conv,
            paddings.len() as i32,
            paddings.as_ptr(),
            strides.as_ptr(),
            dilations.as_ptr(),
            miopenConvolutionMode_t::miopenConvolution,
        )?;
        // Grouped convolutions are expressed by the number of channels of the filter
        let input_channels = plan.problem.x.dimensions[1];
        let filter_channels = plan.problem.w.dimensions[1];
        if filter_channels <= 0 || input_channels % filter_channels != 0 {
            return Err(cudnnError_t::NOT_SUPPORTED);
        }
        let groups = input_channels / filter_channels;
        if groups > 1 {
            miopen.miopenSetConvolutionGroupCount(plan.conv, groups as i32)?;
        }
        let search_workspace = &mut *handle
            .search_workspace
            .lock()
            .map_err(|_| cudnnError_t::INTERNAL_ERROR)?;
        let solutions = search_workspace.search_cache.get_or_insert(
            miopen,
            handle.base,
            plan.problem.direction,
            plan.conv,
            plan.x,
            plan.w,
            plan.y,
        )?;
        // Solutions are sorted from the fastest
        let solution = solutions.first().ok_or(cudnnError_t::NOT_SUPPORTED)?;
        plan.algo = solution.algo.0;
        plan.workspace_size = solution.memory;
        Ok(plan)
    }
}

impl Drop for ConvolutionPlan {
    fn drop(&mut self) {
        let Ok(miopen) = miopen() else {
            return;
        };
        unsafe {
            for desc in [self.x, self.w, self.y] {
                if !desc.0.is_null() {
                    miopen.miopenDestroyTensorDescriptor(desc).ok();
                }
            }
            if !self.conv.0.is_null() {
                miopen.miopenDestroyConvolutionDescriptor(self.conv).ok();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_types::cudnn9::cudnnBackendAttributeName_t as Attr;
    use cuda_types::cudnn9::cudnnBackendAttributeType_t as Type;
    use cuda_types::cudnn9::cudnnBackendDescriptorType_t as Desc;
    use hip_runtime_sys::*;

    unsafe fn create(type_: Desc) -> cudnnBackendDescriptor_t {
        let mut desc = ptr::null_mut();
        backend_create_descriptor(type_, &mut desc).unwrap();
        desc
    }

    unsafe fn set<T>(desc: cudnnBackendDescriptor_t, name: Attr, type_: Type, values: &[T]) {
        backend_set_attribute(
            desc,
            name,
            type_,
            values.len() as i64,
            values.as_ptr().cast(),
        )
        .unwrap();
    }

    unsafe fn get_i64(desc: cudnnBackendDescriptor_t, name: Attr) -> i64 {
        let mut value = 0i64;
        let mut count = 0;
        backend_get_attribute(
            desc,
            name,
            Type::CUDNN_TYPE_INT64,
            1,
            &mut count,
            ptr::from_mut(&mut value).cast(),
        )
        .unwrap();
        assert_eq!(count, 1);
        value
    }

    unsafe fn get_descriptor(
        desc: cudnnBackendDescriptor_t,
        name: Attr,
        type_: Desc,
    ) -> cudnnBackendDescriptor_t {
        let mut result = create(type_);
        let mut count = 0;
        backend_get_attribute(
            desc,
            name,
            Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &mut count,
            ptr::from_mut(&mut result).cast(),
        )
        .unwrap();
        assert_eq!(count, 1);
        result
    }

    unsafe fn tensor(uid: i64, dims: [i64; 4]) -> cudnnBackendDescriptor_t {
        let desc = create(Desc::CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        let strides = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
        set(
            desc,
            Attr::CUDNN_ATTR_TENSOR_UNIQUE_ID,
            Type::CUDNN_TYPE_INT64,
            &[uid],
        );
        set(
            desc,
            Attr::CUDNN_ATTR_TENSOR_DATA_TYPE,
            Type::CUDNN_TYPE_DATA_TYPE,
            &[cudnnDataType_t::CUDNN_DATA_FLOAT],
        );
        set(
            desc,
            Attr::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            Type::CUDNN_TYPE_INT64,
            &[16i64],
        );
        set(
            desc,
            Attr::CUDNN_ATTR_TENSOR_DIMENSIONS,
            Type::CUDNN_TYPE_INT64,
            &dims,
        );
        set(
            desc,
            Attr::CUDNN_ATTR_TENSOR_STRIDES,
            Type::CUDNN_TYPE_INT64,
            &strides,
        );
        set(
            desc,
            Attr::CUDNN_ATTR_TENSOR_IS_VIRTUAL,
            Type::CUDNN_TYPE_BOOLEAN,
            &[false],
        );
        backend_finalize(desc).unwrap();
        desc
    }

    unsafe fn convolution() -> cudnnBackendDescriptor_t {
        let desc = create(Desc::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        set(
            desc,
            Attr::CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
            Type::CUDNN_TYPE_INT64,
            &[2i64],
        );
        set(
            desc,
            Attr::CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
            Type::CUDNN_TYPE_DATA_TYPE,
            &[cudnnDataType_t::CUDNN_DATA_FLOAT],
        );
        set(
            desc,
            Attr::CUDNN_ATTR_CONVOLUTION_CONV_MODE,
            Type::CUDNN_TYPE_CONVOLUTION_MODE,
            &[cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION],
        );
        for name in [
            Attr::CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
            Attr::CUDNN_ATTR_CONVOLUTION_DILATIONS,
        ] {
            set(desc, name, Type::CUDNN_TYPE_INT64, &[1i64, 1]);
        }
        for name in [
            Attr::CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
            Attr::CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
        ] {
            set(desc, name, Type::CUDNN_TYPE_INT64, &[0i64, 0]);
        }
        backend_finalize(desc).unwrap();
        desc
    }

    // x: 1x1x4x4, w: 1x1x3x3, y: 1x1x2x2
    const X_DIMS: [i64; 4] = [1, 1, 4, 4];
    const W_DIMS: [i64; 4] = [1, 1, 3, 3];
    const Y_DIMS: [i64; 4] = [1, 1, 2, 2];

    // Builds a single operation graph, like MXNet does
    unsafe fn graph(handle: cudnnHandle_t, op_type: Desc) -> cudnnBackendDescriptor_t {
        let conv = convolution();
        let (names, uids_dims): ([Attr; 6], [(i64, [i64; 4]); 3]) = match op_type {
            Desc::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR => (
                [
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                ],
                [(1, X_DIMS), (2, W_DIMS), (3, Y_DIMS)],
            ),
            Desc::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR => (
                [
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                ],
                [(1, W_DIMS), (2, Y_DIMS), (3, X_DIMS)],
            ),
            _ => (
                [
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                ],
                [(1, X_DIMS), (2, Y_DIMS), (3, W_DIMS)],
            ),
        };
        let tensors = uids_dims.map(|(uid, dims)| tensor(uid, dims));
        let op = create(op_type);
        set(op, names[0], Type::CUDNN_TYPE_BACKEND_DESCRIPTOR, &[conv]);
        for (name, tensor) in names[1..4].iter().zip(tensors) {
            set(op, *name, Type::CUDNN_TYPE_BACKEND_DESCRIPTOR, &[tensor]);
        }
        set(op, names[4], Type::CUDNN_TYPE_FLOAT, &[1.0f32]);
        set(op, names[5], Type::CUDNN_TYPE_FLOAT, &[0.0f32]);
        backend_finalize(op).unwrap();
        let graph = create(Desc::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set(
            graph,
            Attr::CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
            Type::CUDNN_TYPE_HANDLE,
            &[handle],
        );
        set(
            graph,
            Attr::CUDNN_ATTR_OPERATIONGRAPH_OPS,
            Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[op],
        );
        backend_finalize(graph).unwrap();
        // The graph captures its operations by value, everything else can be destroyed
        for desc in tensors.into_iter().chain([conv, op]) {
            backend_destroy_descriptor(desc).unwrap();
        }
        graph
    }

    unsafe fn heuristic_configs(graph: cudnnBackendDescriptor_t) -> Vec<cudnnBackendDescriptor_t> {
        let heur = create(Desc::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        set(
            heur,
            Attr::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
            Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[graph],
        );
        set(
            heur,
            Attr::CUDNN_ATTR_ENGINEHEUR_MODE,
            Type::CUDNN_TYPE_HEUR_MODE,
            &[cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_B],
        );
        backend_finalize(heur).unwrap();
        let mut count = 0;
        backend_get_attribute(
            heur,
            Attr::CUDNN_ATTR_ENGINEHEUR_RESULTS,
            Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            0,
            &mut count,
            ptr::null_mut(),
        )
        .unwrap();
        let mut configs = (0..count)
            .map(|_| create(Desc::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR))
            .collect::<Vec<_>>();
        backend_get_attribute(
            heur,
            Attr::CUDNN_ATTR_ENGINEHEUR_RESULTS,
            Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            count,
            &mut count,
            configs.as_mut_ptr().cast(),
        )
        .unwrap();
        backend_destroy_descriptor(heur).unwrap();
        configs
    }

    struct Handle(cudnnHandle_t);

    impl Handle {
        fn new() -> Self {
            let mut handle = ptr::null_mut();
            unsafe { crate::r#impl::create(&mut handle) }.unwrap();
            Self(handle)
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            crate::r#impl::destroy(self.0).unwrap();
        }
    }

    #[test]
    fn get_attribute_reports_count_and_copies_values() {
        unsafe {
            let desc = tensor(7, X_DIMS);
            assert_eq!(get_i64(desc, Attr::CUDNN_ATTR_TENSOR_UNIQUE_ID), 7);
            let mut dims = [0i64; 8];
            let mut count = 0;
            backend_get_attribute(
                desc,
                Attr::CUDNN_ATTR_TENSOR_DIMENSIONS,
                Type::CUDNN_TYPE_INT64,
                dims.len() as i64,
                &mut count,
                dims.as_mut_ptr().cast(),
            )
            .unwrap();
            assert_eq!(count, 4);
            assert_eq!(dims[..4], X_DIMS);
            // Mismatched type
            assert_eq!(
                backend_get_attribute(
                    desc,
                    Attr::CUDNN_ATTR_TENSOR_DIMENSIONS,
                    Type::CUDNN_TYPE_FLOAT,
                    1,
                    &mut count,
                    dims.as_mut_ptr().cast(),
                ),
                Err(cudnnError_t::BAD_PARAM)
            );
            // Finalized descriptors can't be modified
            assert_eq!(
                backend_set_attribute(
                    desc,
                    Attr::CUDNN_ATTR_TENSOR_UNIQUE_ID,
                    Type::CUDNN_TYPE_INT64,
                    1,
                    ptr::from_ref(&1i64).cast(),
                ),
                Err(cudnnError_t::NOT_SUPPORTED)
            );
            backend_destroy_descriptor(desc).unwrap();
        }
    }

    #[test]
    fn unfinalized_descriptor_can_not_be_referenced() {
        unsafe {
            let tensor = create(Desc::CUDNN_BACKEND_TENSOR_DESCRIPTOR);
            let op = create(Desc::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
            assert_eq!(
                backend_set_attribute(
                    op,
                    Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                    Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    ptr::from_ref(&tensor).cast(),
                ),
                Err(cudnnError_t::BAD_PARAM)
            );
            backend_destroy_descriptor(op).unwrap();
            backend_destroy_descriptor(tensor).unwrap();
        }
    }

    #[test]
    fn heuristics_return_single_engine_without_notes_or_knobs() {
        let handle = Handle::new();
        unsafe {
            let graph = graph(
                handle.0,
                Desc::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            );
            let configs = heuristic_configs(graph);
            assert_eq!(configs.len(), 1);
            let engine = get_descriptor(
                configs[0],
                Attr::CUDNN_ATTR_ENGINECFG_ENGINE,
                Desc::CUDNN_BACKEND_ENGINE_DESCRIPTOR,
            );
            assert_eq!(get_i64(engine, Attr::CUDNN_ATTR_ENGINE_GLOBAL_INDEX), 0);
            let mut notes = [cudnnBackendNumericalNote_t(0); 10];
            let mut count = -1;
            backend_get_attribute(
                engine,
                Attr::CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                Type::CUDNN_TYPE_NUMERICAL_NOTE,
                notes.len() as i64,
                &mut count,
                notes.as_mut_ptr().cast(),
            )
            .unwrap();
            assert_eq!(count, 0);
            let mut knobs = [create(Desc::CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR)];
            backend_get_attribute(
                configs[0],
                Attr::CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &mut count,
                knobs.as_mut_ptr().cast(),
            )
            .unwrap();
            assert_eq!(count, 0);
            // Fallback engines other than 0 don't exist
            let fallback = create(Desc::CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            set(
                fallback,
                Attr::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                &[graph],
            );
            set(
                fallback,
                Attr::CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                Type::CUDNN_TYPE_INT64,
                &[1i64],
            );
            assert_eq!(backend_finalize(fallback), Err(cudnnError_t::NOT_SUPPORTED));
            for desc in [graph, engine, configs[0], knobs[0], fallback] {
                backend_destroy_descriptor(desc).unwrap();
            }
        }
    }

    unsafe fn to_device(data: &[f32]) -> *mut c_void {
        let mut ptr = ptr::null_mut();
        hipMalloc(&mut ptr, data.len() * 4).unwrap();
        hipMemcpy(
            ptr,
            data.as_ptr().cast(),
            data.len() * 4,
            hipMemcpyKind::hipMemcpyHostToDevice,
        )
        .unwrap();
        ptr
    }

    unsafe fn to_host(ptr: *mut c_void, len: usize) -> Vec<f32> {
        let mut data = vec![0f32; len];
        hipMemcpy(
            data.as_mut_ptr().cast(),
            ptr,
            len * 4,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        )
        .unwrap();
        hipFree(ptr).unwrap();
        data
    }

    // Runs a 4x4 * 3x3 convolution graph. Inputs and the output are given in the order of uids
    unsafe fn execute(op_type: Desc, inputs: [&[f32]; 2], output_len: usize) -> Vec<f32> {
        let handle = Handle::new();
        let graph = graph(handle.0, op_type);
        let configs = heuristic_configs(graph);
        let plan = create(Desc::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        set(
            plan,
            Attr::CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
            Type::CUDNN_TYPE_HANDLE,
            &[handle.0],
        );
        set(
            plan,
            Attr::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            Type::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[configs[0]],
        );
        backend_finalize(plan).unwrap();
        let workspace_size = get_i64(plan, Attr::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
        let mut workspace = ptr::null_mut();
        if workspace_size > 0 {
            hipMalloc(&mut workspace, workspace_size as usize).unwrap();
        }
        let pointers = [
            to_device(inputs[0]),
            to_device(inputs[1]),
            to_device(&vec![0f32; output_len]),
        ];
        let variant_pack = create(Desc::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        set(
            variant_pack,
            Attr::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
            Type::CUDNN_TYPE_INT64,
            &[1i64, 2, 3],
        );
        set(
            variant_pack,
            Attr::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
            Type::CUDNN_TYPE_VOID_PTR,
            &pointers,
        );
        set(
            variant_pack,
            Attr::CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
            Type::CUDNN_TYPE_VOID_PTR,
            &[workspace],
        );
        backend_finalize(variant_pack).unwrap();
        let context: &Context = FromCuda::<_, cudnnError_t>::from_cuda(&handle.0).unwrap();
        backend_execute(context, plan, variant_pack).unwrap();
        hipDeviceSynchronize().unwrap();
        let result = to_host(pointers[2], output_len);
        hipFree(pointers[0]).unwrap();
        hipFree(pointers[1]).unwrap();
        if !workspace.is_null() {
            hipFree(workspace).unwrap();
        }
        for desc in [variant_pack, plan, configs[0], graph] {
            backend_destroy_descriptor(desc).unwrap();
        }
        result
    }

    const X: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    const W: [f32; 9] = [1., 0., -1., 2., 0., -2., 1., 0., -1.];
    const DY: [f32; 4] = [1., 2., 3., 4.];

    fn reference_forward() -> Vec<f32> {
        let mut y = vec![0f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                for ki in 0..3 {
                    for kj in 0..3 {
                        y[i * 2 + j] += X[(i + ki) * 4 + j + kj] * W[ki * 3 + kj];
                    }
                }
            }
        }
        y
    }

    fn reference_backward() -> (Vec<f32>, Vec<f32>) {
        let mut dx = vec![0f32; 16];
        let mut dw = vec![0f32; 9];
        for i in 0..2 {
            for j in 0..2 {
                for ki in 0..3 {
                    for kj in 0..3 {
                        dx[(i + ki) * 4 + j + kj] += DY[i * 2 + j] * W[ki * 3 + kj];
                        dw[ki * 3 + kj] += DY[i * 2 + j] * X[(i + ki) * 4 + j + kj];
                    }
                }
            }
        }
        (dx, dw)
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-4, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn execute_convolution_forward() {
        let y = unsafe {
            execute(
                Desc::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                [&X, &W],
                4,
            )
        };
        assert_close(&y, &reference_forward());
    }

    #[test]
    fn execute_convolution_backward_data() {
        let dx = unsafe {
            execute(
                Desc::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                [&W, &DY],
                16,
            )
        };
        assert_close(&dx, &reference_backward().0);
    }

    #[test]
    fn execute_convolution_backward_filter() {
        let dw = unsafe {
            execute(
                Desc::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                [&X, &DY],
                9,
            )
        };
        assert_close(&dw, &reference_backward().1);
    }
}
