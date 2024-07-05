use std::collections::VecDeque;
use std::sync::Arc;
use arrow::array::{Array, ArrayData, ArrayRef, MapArray, StructArray};
use arrow::datatypes::{DataType, Field};
use arrow_buffer::{Buffer, ToByteSlice};
use datafusion_common::exec_err;
use datafusion_common::Result;
use datafusion_expr::Signature;

fn make_map(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() % 2 != 0 {
        return exec_err!("map requires an even number of arguments, got {} instead", args.len());
    }

    let key_field = Arc::new(Field::new("key", args[0].data_type().clone(), args[0].null_count() > 0));
    let value_field = Arc::new(Field::new("value", args[1].data_type().clone(), args[1].null_count() > 0));
    let mut entry_struct_buffer = VecDeque::new();
    let mut entry_offsets_buffer = VecDeque::new();
    entry_offsets_buffer.push_back(0);

    for key_value in args.chunks_exact(2) {
        let key = &key_value[0];
        let value = &key_value[1];
        if key.data_type() != args[0].data_type() {
            return exec_err!("map key type must be consistent, got {:?} and {:?}", key.data_type(), args[0].data_type());
        }
        if value.data_type() != args[1].data_type() {
            return exec_err!("map value type must be consistent, got {:?} and {:?}", value.data_type(), args[1].data_type());
        }
        if key.len() != value.len() {
            return exec_err!("map key and value must have the same length, got {} and {}", key.len(), value.len());
        }

        entry_struct_buffer.push_back((Arc::clone(&key_field), Arc::clone(key)));
        entry_struct_buffer.push_back((Arc::clone(&value_field), Arc::clone(value)));
        entry_offsets_buffer.push_back(key.len() as u32);
    }
    let entry_struct: Vec<(Arc<Field>, ArrayRef)> = entry_struct_buffer.into();
    let entry_struct = StructArray::from(entry_struct);

    let map_data_type = DataType::Map(
        Arc::new(Field::new("entries",
           entry_struct.data_type().clone(), false)),
          false);

    let entry_offsets: Vec<u32> = entry_offsets_buffer.into();
    let entry_offsets_buffer = Buffer::from(entry_offsets.to_byte_slice());

    let map_data = ArrayData::builder(map_data_type)
        .len(entry_offsets.len() - 1)
        .add_buffer(entry_offsets_buffer)
        .add_child_data(entry_struct.to_data())
        .build()?;

    Ok(Arc::new(MapArray::from(map_data)))
}

pub struct MapFunc {
    signature: Signature,
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use arrow::array::{AsArray, Int64Array};
    use datafusion_common::{Result, ScalarValue};
    use crate::core::map::make_map;

    #[test]
    fn test_create_map() -> Result<()> {
        let key = Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3)]));
        let value = Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30)]));
        let map = make_map(&[key, value])?;
        let value = ScalarValue::Map(Arc::new(map.as_map().to_owned()));
        let display = format!("{}", value);
        assert_eq!(display, "[{1:10,2:20,3:30}]");
        Ok(())
    }

    // add tests
}