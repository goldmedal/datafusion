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

use datafusion_expr::Expr;
use datafusion_expr::expr::ScalarFunction;
use datafusion_functions::core::map::{map_one_udf, map_udf};
use crate::make_array::make_array;

pub fn map(keys: Vec<Expr>, values: Vec<Expr>) -> Expr {
    let keys = make_array(keys);
    let values = make_array(values);
    Expr::ScalarFunction(ScalarFunction::new_udf(
        map_udf(),
        vec![keys, values],
    ))
}

pub fn map_from_array(args: Vec<Expr>) -> Expr {
    Expr::ScalarFunction(ScalarFunction::new_udf(
        map_one_udf(),
        args,
    ))
}
