// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::{ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl};
use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::internal_err;
use datafusion_expr_common::columnar_value::ColumnarValue;
use datafusion_expr_common::signature::Signature;
use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

/// A scalar UDF that can invoke using async methods
///
/// Note this is less efficient than the ScalarUDFImpl, but it can be used
/// to register remote functions in the context.
///
/// The name is chosen to  mirror ScalarUDFImpl
#[async_trait]
pub trait AsyncScalarUDFImpl: Debug + Send + Sync {
    /// the function cast as any
    fn as_any(&self) -> &dyn Any;

    /// The name of the function
    fn name(&self) -> &str;

    /// The signature of the function
    fn signature(&self) -> &Signature;

    /// The return type of the function
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType>;

    /// The ideal batch size for this function.
    fn ideal_batch_size(&self) -> Option<usize> {
        None
    }

    /// Invoke the function asynchronously with the async arguments
    async fn invoke_async_with_args(
        &self,
        args: AsyncScalarFunctionArgs,
        option: &ConfigOptions,
    ) -> Result<ArrayRef>;
}

/// A scalar UDF that must be invoked using async methods
///
/// Note this is not meant to be used directly, but is meant to be an implementation detail
/// for AsyncUDFImpl.
///
/// This is used to register remote functions in the context. The function
/// should not be invoked by DataFusion. It's only used to generate the logical
/// plan and unparsed them to SQL.
#[derive(Debug)]
pub struct AsyncScalarUDF {
    inner: Arc<dyn AsyncScalarUDFImpl>,
}

impl AsyncScalarUDF {
    pub fn new(inner: Arc<dyn AsyncScalarUDFImpl>) -> Self {
        Self { inner }
    }

    /// The ideal batch size for this function
    pub fn ideal_batch_size(&self) -> Option<usize> {
        self.inner.ideal_batch_size()
    }

    /// Turn this AsyncUDF into a ScalarUDF, suitable for
    /// registering in the context
    pub fn into_scalar_udf(self) -> Arc<ScalarUDF> {
        Arc::new(ScalarUDF::new_from_impl(self))
    }

    /// Invoke the function asynchronously with the async arguments
    pub async fn invoke_async_with_args(
        &self,
        args: AsyncScalarFunctionArgs,
        option: &ConfigOptions,
    ) -> Result<ArrayRef> {
        self.inner.invoke_async_with_args(args, option).await
    }
}

impl ScalarUDFImpl for AsyncScalarUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn signature(&self) -> &Signature {
        self.inner.signature()
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        self.inner.return_type(_arg_types)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("async functions should not be called directly")
    }
}

impl Display for AsyncScalarUDF {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "AsyncScalarUDF: {}", self.inner.name())
    }
}

#[derive(Debug)]
pub struct AsyncScalarFunctionArgs {
    pub args: Vec<ColumnarValue>,
    pub number_rows: usize,
    pub schema: SchemaRef,
}
