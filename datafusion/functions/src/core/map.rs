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

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{Array, ArrayData, ArrayRef, FixedSizeListArray, LargeListArray, ListArray, MapArray, StructArray};
use arrow::datatypes::{DataType, Field, SchemaBuilder};
use arrow::datatypes::DataType::{Int32, Utf8};
use arrow_buffer::{Buffer, ToByteSlice};

use datafusion_common::{exec_err, ScalarValue};
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};

make_udf_function!(MapFunc, MAP, map_udf);
make_udf_function!(MapOneFunc, MAP_ONE, map_one_udf);

/// Check if we can evaluate the expr to constant directly.
///
/// # Example
/// ```sql
/// SELECT make_map('type', 'test') from test
/// ```
/// We can evaluate the result of `make_map` directly.
fn can_evaluate_to_const(args: &[ColumnarValue]) -> bool {
    args.iter()
        .all(|arg| matches!(arg, ColumnarValue::Scalar(_)))
}

fn get_scalar_from_col(c: &ColumnarValue) -> ScalarValue {
    match c {
        ColumnarValue::Scalar(s) => s.clone(),
        _ => todo!(""),
    }
}

fn make_map_batch_one_args(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    if args.len() % 2 != 0 {
        return exec_err!(
            "make_map requires exactly 2 arguments, got {} instead",
            args.len()
        );
    }

    let len = args.len() / 2;
    let key_iter = args[0..len].iter().map(get_scalar_from_col);
    let key = ScalarValue::iter_to_array(key_iter)?;
    let val_iter = args[len..].iter().map(get_scalar_from_col);
    let value = ScalarValue::iter_to_array(val_iter)?;

    let key = get_first_element(key);
    let value = get_first_element(value);
    let can_evaluate_to_const = can_evaluate_to_const(args);
    make_map_batch_internal(key, value, can_evaluate_to_const)
}

fn get_first_element(value: ArrayRef) -> ArrayRef {
    match value.data_type() {
        DataType::List(_) => {
            let list_array = value.as_any().downcast_ref::<ListArray>().unwrap();
            list_array.value(0)
        }
        DataType::LargeList(_) => {
            let list_array = value.as_any().downcast_ref::<LargeListArray>().unwrap();
            list_array.value(0)
        }
        DataType::FixedSizeList(_, _) => {
            let list_array = value.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            list_array.value(0)
        }
        _ => value,
    }
}

fn make_map_batch(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    if args.len() != 2 {
        return exec_err!(
            "make_map requires exactly 2 arguments, got {} instead",
            args.len()
        );
    }

    let can_evaluate_to_const = can_evaluate_to_const(args);

    let key = get_first_array_ref(&args[0])?;
    let value = get_first_array_ref(&args[1])?;
    make_map_batch_internal(key, value, can_evaluate_to_const)
}

fn get_first_array_ref(columnar_value: &ColumnarValue) -> Result<ArrayRef> {
    match columnar_value {
        ColumnarValue::Scalar(value) => match value {
            ScalarValue::List(array) => Ok(array.value(0)),
            ScalarValue::LargeList(array) => Ok(array.value(0)),
            ScalarValue::FixedSizeList(array) => Ok(array.value(0)),
            _ => exec_err!("Expected array, got {:?}", value),
        },
        ColumnarValue::Array(array) => exec_err!("Expected scalar, got {:?}", array),
    }
}

fn make_map_batch_internal(
    keys: ArrayRef,
    values: ArrayRef,
    can_evaluate_to_const: bool,
) -> Result<ColumnarValue> {
    if keys.null_count() > 0 {
        return exec_err!("map key cannot be null");
    }

    if keys.len() != values.len() {
        return exec_err!("map requires key and value lists to have the same length");
    }

    let key_field = Arc::new(Field::new("key", keys.data_type().clone(), false));
    let value_field = Arc::new(Field::new("value", values.data_type().clone(), true));
    let mut entry_struct_buffer: VecDeque<(Arc<Field>, ArrayRef)> = VecDeque::new();
    let mut entry_offsets_buffer = VecDeque::new();
    entry_offsets_buffer.push_back(0);

    entry_struct_buffer.push_back((Arc::clone(&key_field), Arc::clone(&keys)));
    entry_struct_buffer.push_back((Arc::clone(&value_field), Arc::clone(&values)));
    entry_offsets_buffer.push_back(keys.len() as u32);

    let entry_struct: Vec<(Arc<Field>, ArrayRef)> = entry_struct_buffer.into();
    let entry_struct = StructArray::from(entry_struct);

    let map_data_type = DataType::Map(
        Arc::new(Field::new(
            "entries",
            entry_struct.data_type().clone(),
            false,
        )),
        false,
    );

    let entry_offsets: Vec<u32> = entry_offsets_buffer.into();
    let entry_offsets_buffer = Buffer::from(entry_offsets.to_byte_slice());

    let map_data = ArrayData::builder(map_data_type)
        .len(entry_offsets.len() - 1)
        .add_buffer(entry_offsets_buffer)
        .add_child_data(entry_struct.to_data())
        .build()?;
    let map_array = Arc::new(MapArray::from(map_data));

    Ok(if can_evaluate_to_const {
        ColumnarValue::Scalar(ScalarValue::try_from_array(map_array.as_ref(), 0)?)
    } else {
        ColumnarValue::Array(map_array)
    })
}

#[derive(Debug)]
pub struct MapFunc {
    signature: Signature,
}

impl Default for MapFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl MapFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for MapFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "map"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.len() % 2 != 0 {
            return exec_err!(
                "map requires an even number of arguments, got {} instead",
                arg_types.len()
            );
        }
        let mut builder = SchemaBuilder::new();
        builder.push(Field::new(
            "key",
            get_element_type(&arg_types[0])?.clone(),
            false,
        ));
        builder.push(Field::new(
            "value",
            get_element_type(&arg_types[1])?.clone(),
            true,
        ));
        let fields = builder.finish().fields;
        Ok(DataType::Map(
            Arc::new(Field::new("entries", DataType::Struct(fields), false)),
            false,
        ))
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_map_batch(args)
    }
}

fn get_element_type(data_type: &DataType) -> Result<&DataType> {
    match data_type {
        DataType::List(element) => Ok(element.data_type()),
        DataType::LargeList(element) => Ok(element.data_type()),
        DataType::FixedSizeList(element, _) => Ok(element.data_type()),
        _ => exec_err!(
            "Expected list, large_list or fixed_size_list, got {:?}",
            data_type
        ),
    }
}

#[derive(Debug)]
pub struct MapOneFunc {
    signature: Signature,
}

impl Default for MapOneFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl MapOneFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for MapOneFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "map_one"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.len() == 1 {
            return exec_err!(
                "map_one only accepts 1 arguments, got {} instead",
                arg_types.len()
            );
        }

        // let key_type = &arg_types[0];
        // let val_type = arg_types.last().unwrap();
        //
        let mut builder = SchemaBuilder::new();
        // TODO: get the correct type
        builder.push(Field::new(
            "key",
            Utf8,
            false,
        ));
        builder.push(Field::new(
            "value",
           Int32,
            true,
        ));
        let fields = builder.finish().fields;
        Ok(DataType::Map(
            Arc::new(Field::new("entries", DataType::Struct(fields), false)),
            false,
        ))
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_map_batch_one_args(args)
    }
}
