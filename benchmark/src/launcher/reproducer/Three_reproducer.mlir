module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c23 = arith.constant 23 : index
    %c22 = arith.constant 22 : index
    %c21 = arith.constant 21 : index
    %c20 = arith.constant 20 : index
    %c19 = arith.constant 19 : index
    %c18 = arith.constant 18 : index
    %c17 = arith.constant 17 : index
    %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<8x8xf32>
    %c7_i32 = arith.constant 7 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c7_i32 : i32
    %2 = arith.divsi %1, %c8_i32 : i32
    %3 = arith.addi %arg4, %c7_i32 : i32
    %4 = arith.divsi %3, %c8_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c8_i32 : i32
    %15 = arith.muli %13, %c8_i32 : i32
    %16 = arith.extsi %arg3 : i32 to i64
    %17 = arith.extsi %arg5 : i32 to i64
    %18 = arith.extsi %arg6 : i32 to i64
    %19 = tt.make_tensor_ptr %arg0, [%16, %17], [%18, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    %22 = tt.make_tensor_ptr %arg1, [%17, %20], [%21, %c1_i64], [%c0_i32, %15] {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    %23 = arith.divsi %arg5, %c8_i32 : i32
    %24:10 = scf.for %arg9 = %c0_i32 to %23 step %c1_i32 iter_args(%arg10 = %19, %arg11 = %22, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst) -> (!tt.ptr<tensor<8x8xf32>>, !tt.ptr<tensor<8x8xf32>>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)  : i32 {
      %37 = triton_cpu.extract_memref %arg10 : <tensor<8x8xf32>> -> memref<?x?xf32, strided<[?, 1]>>
      %38:2 = triton_cpu.extract_indices %arg10 : <tensor<8x8xf32>> -> index, index
      %39 = triton_cpu.extract_memref %arg11 : <tensor<8x8xf32>> -> memref<?x?xf32, strided<[?, 1]>>
      %40:2 = triton_cpu.extract_indices %arg11 : <tensor<8x8xf32>> -> index, index
      %41 = arith.addi %38#1, %c16 : index
      %42 = arith.addi %40#0, %c16 : index
      %43 = vector.load %39[%40#0, %40#1] : memref<?x?xf32, strided<[?, 1]>>, vector<8xf32>
      %44 = arith.addi %40#0, %c1 : index
      %45 = vector.load %39[%44, %40#1] : memref<?x?xf32, strided<[?, 1]>>, vector<8xf32>
      memref.prefetch %39[%42, %40#1], read, locality<1>, data : memref<?x?xf32, strided<[?, 1]>>
      %46 = memref.load %37[%38#0, %38#1] : memref<?x?xf32, strided<[?, 1]>>
      %47 = vector.broadcast %46 : f32 to vector<8xf32>
      %48 = arith.addi %38#0, %c1 : index
      %49 = memref.load %37[%48, %38#1] : memref<?x?xf32, strided<[?, 1]>>
      %50 = vector.broadcast %49 : f32 to vector<8xf32>
      memref.prefetch %37[%38#0, %41], read, locality<3>, data : memref<?x?xf32, strided<[?, 1]>>
      %51 = vector.fma %43, %47, %arg12 : vector<8xf32>
      %52 = arith.addi %38#0, %c2 : index
      %53 = memref.load %37[%52, %38#1] : memref<?x?xf32, strided<[?, 1]>>
      %54 = vector.broadcast %53 : f32 to vector<8xf32>
      memref.prefetch %37[%48, %41], read, locality<3>, data : memref<?x?xf32, strided<[?, 1]>>
      %55 = vector.fma %43, %50, %arg13 : vector<8xf32>
      %56 = arith.addi %38#0, %c3 : index
      %57 = memref.load %37[%56, %38#1] : memref<?x?xf32, strided<[?, 1]>>
      %58 = vector.broadcast %57 : f32 to vector<8xf32>
      memref.prefetch %37[%52, %41], read, locality<3>, data : memref<?x?xf32, strided<[?, 1]>>
      %59 = vector.fma %43, %54, %arg14 : vector<8xf32>
      %60 = arith.addi %38#0, %c4 : index
      %61 = memref.load %37[%60, %38#1] : memref<?x?xf32, strided<[?, 1]>>
      %62 = vector.broadcast %61 : f32 to vector<8xf32>
      memref.prefetch %37[%56, %41], read, locality<3>, data : memref<?x?xf32, strided<[?, 1]>>
      %63 = vector.fma %43, %58, %arg15 : vector<8xf32>
      %64 = arith.addi %38#0, %c5 : index
      %65 = memref.load %37[%64, %38#1] : memref<?x?xf32, strided<[?, 1]>>
      %66 = vector.broadcast %65 : f32 to vector<8xf32>
      memref.prefetch %37[%60, %41], read, locality<3>, data : memref<?x?xf32, strided<[?, 1]>>
      %67 = vector.fma %43, %62, %arg16 : vector<8xf32>
      %68 = arith.addi %38#0, %c6 : index
      %69 = memref.load %37[%68, %38#1] : memref<?x?xf32, strided<[?, 1]>>
      %70 = vector.broadcast %69 : f32 to vector<8xf32>
      memref.prefetch %37[%64, %41], read, locality<3>, data : memref<?x?xf32, strided<[?, 1]>>
      %71 = vector.fma %43, %66, %arg17 : vector<8xf32>
      %72 = arith.addi %38#0, %c7 : index
      %73 = memref.load %37[%72, %38#1] : memref<?x?xf32, strided<[?, 1]>>
      %74 = vector.broadcast %73 : f32 to vector<8xf32>
      memref.prefetch %37[%68, %41], read, locality<3>, data : memref<?x?xf32, strided<[?, 1]>>
      %75 = vector.fma %43, %70, %arg18 : vector<8xf32>
      memref.prefetch %37[%72, %41], read, locality<3>, data : memref<?x?xf32, strided<[?, 1]>>
      %76 = vector.fma %43, %74, %arg19 : vector<8xf32>
      %77 = arith.addi %40#0, %c2 : index
      %78 = vector.load %39[%77, %40#1] : memref<?x?xf32, strided<[?, 1]>>, vector<8xf32>
      %79 = arith.addi %40#0, %c17 : index
      memref.prefetch %39[%79, %40#1], read, locality<1>, data : memref<?x?xf32, strided<[?, 1]>>
      %80 = arith.addi %38#1, %c1 : index
      %81 = memref.load %37[%38#0, %80] : memref<?x?xf32, strided<[?, 1]>>
      %82 = vector.broadcast %81 : f32 to vector<8xf32>
      %83 = memref.load %37[%48, %80] : memref<?x?xf32, strided<[?, 1]>>
      %84 = vector.broadcast %83 : f32 to vector<8xf32>
      %85 = vector.fma %45, %82, %51 : vector<8xf32>
      %86 = memref.load %37[%52, %80] : memref<?x?xf32, strided<[?, 1]>>
      %87 = vector.broadcast %86 : f32 to vector<8xf32>
      %88 = vector.fma %45, %84, %55 : vector<8xf32>
      %89 = memref.load %37[%56, %80] : memref<?x?xf32, strided<[?, 1]>>
      %90 = vector.broadcast %89 : f32 to vector<8xf32>
      %91 = vector.fma %45, %87, %59 : vector<8xf32>
      %92 = memref.load %37[%60, %80] : memref<?x?xf32, strided<[?, 1]>>
      %93 = vector.broadcast %92 : f32 to vector<8xf32>
      %94 = vector.fma %45, %90, %63 : vector<8xf32>
      %95 = memref.load %37[%64, %80] : memref<?x?xf32, strided<[?, 1]>>
      %96 = vector.broadcast %95 : f32 to vector<8xf32>
      %97 = vector.fma %45, %93, %67 : vector<8xf32>
      %98 = memref.load %37[%68, %80] : memref<?x?xf32, strided<[?, 1]>>
      %99 = vector.broadcast %98 : f32 to vector<8xf32>
      %100 = vector.fma %45, %96, %71 : vector<8xf32>
      %101 = memref.load %37[%72, %80] : memref<?x?xf32, strided<[?, 1]>>
      %102 = vector.broadcast %101 : f32 to vector<8xf32>
      %103 = vector.fma %45, %99, %75 : vector<8xf32>
      %104 = vector.fma %45, %102, %76 : vector<8xf32>
      %105 = arith.addi %40#0, %c3 : index
      %106 = vector.load %39[%105, %40#1] : memref<?x?xf32, strided<[?, 1]>>, vector<8xf32>
      %107 = arith.addi %40#0, %c18 : index
      memref.prefetch %39[%107, %40#1], read, locality<1>, data : memref<?x?xf32, strided<[?, 1]>>
      %108 = arith.addi %38#1, %c2 : index
      %109 = memref.load %37[%38#0, %108] : memref<?x?xf32, strided<[?, 1]>>
      %110 = vector.broadcast %109 : f32 to vector<8xf32>
      %111 = memref.load %37[%48, %108] : memref<?x?xf32, strided<[?, 1]>>
      %112 = vector.broadcast %111 : f32 to vector<8xf32>
      %113 = vector.fma %78, %110, %85 : vector<8xf32>
      %114 = memref.load %37[%52, %108] : memref<?x?xf32, strided<[?, 1]>>
      %115 = vector.broadcast %114 : f32 to vector<8xf32>
      %116 = vector.fma %78, %112, %88 : vector<8xf32>
      %117 = memref.load %37[%56, %108] : memref<?x?xf32, strided<[?, 1]>>
      %118 = vector.broadcast %117 : f32 to vector<8xf32>
      %119 = vector.fma %78, %115, %91 : vector<8xf32>
      %120 = memref.load %37[%60, %108] : memref<?x?xf32, strided<[?, 1]>>
      %121 = vector.broadcast %120 : f32 to vector<8xf32>
      %122 = vector.fma %78, %118, %94 : vector<8xf32>
      %123 = memref.load %37[%64, %108] : memref<?x?xf32, strided<[?, 1]>>
      %124 = vector.broadcast %123 : f32 to vector<8xf32>
      %125 = vector.fma %78, %121, %97 : vector<8xf32>
      %126 = memref.load %37[%68, %108] : memref<?x?xf32, strided<[?, 1]>>
      %127 = vector.broadcast %126 : f32 to vector<8xf32>
      %128 = vector.fma %78, %124, %100 : vector<8xf32>
      %129 = memref.load %37[%72, %108] : memref<?x?xf32, strided<[?, 1]>>
      %130 = vector.broadcast %129 : f32 to vector<8xf32>
      %131 = vector.fma %78, %127, %103 : vector<8xf32>
      %132 = vector.fma %78, %130, %104 : vector<8xf32>
      %133 = arith.addi %40#0, %c4 : index
      %134 = vector.load %39[%133, %40#1] : memref<?x?xf32, strided<[?, 1]>>, vector<8xf32>
      %135 = arith.addi %40#0, %c19 : index
      memref.prefetch %39[%135, %40#1], read, locality<1>, data : memref<?x?xf32, strided<[?, 1]>>
      %136 = arith.addi %38#1, %c3 : index
      %137 = memref.load %37[%38#0, %136] : memref<?x?xf32, strided<[?, 1]>>
      %138 = vector.broadcast %137 : f32 to vector<8xf32>
      %139 = memref.load %37[%48, %136] : memref<?x?xf32, strided<[?, 1]>>
      %140 = vector.broadcast %139 : f32 to vector<8xf32>
      %141 = vector.fma %106, %138, %113 : vector<8xf32>
      %142 = memref.load %37[%52, %136] : memref<?x?xf32, strided<[?, 1]>>
      %143 = vector.broadcast %142 : f32 to vector<8xf32>
      %144 = vector.fma %106, %140, %116 : vector<8xf32>
      %145 = memref.load %37[%56, %136] : memref<?x?xf32, strided<[?, 1]>>
      %146 = vector.broadcast %145 : f32 to vector<8xf32>
      %147 = vector.fma %106, %143, %119 : vector<8xf32>
      %148 = memref.load %37[%60, %136] : memref<?x?xf32, strided<[?, 1]>>
      %149 = vector.broadcast %148 : f32 to vector<8xf32>
      %150 = vector.fma %106, %146, %122 : vector<8xf32>
      %151 = memref.load %37[%64, %136] : memref<?x?xf32, strided<[?, 1]>>
      %152 = vector.broadcast %151 : f32 to vector<8xf32>
      %153 = vector.fma %106, %149, %125 : vector<8xf32>
      %154 = memref.load %37[%68, %136] : memref<?x?xf32, strided<[?, 1]>>
      %155 = vector.broadcast %154 : f32 to vector<8xf32>
      %156 = vector.fma %106, %152, %128 : vector<8xf32>
      %157 = memref.load %37[%72, %136] : memref<?x?xf32, strided<[?, 1]>>
      %158 = vector.broadcast %157 : f32 to vector<8xf32>
      %159 = vector.fma %106, %155, %131 : vector<8xf32>
      %160 = vector.fma %106, %158, %132 : vector<8xf32>
      %161 = arith.addi %40#0, %c5 : index
      %162 = vector.load %39[%161, %40#1] : memref<?x?xf32, strided<[?, 1]>>, vector<8xf32>
      %163 = arith.addi %40#0, %c20 : index
      memref.prefetch %39[%163, %40#1], read, locality<1>, data : memref<?x?xf32, strided<[?, 1]>>
      %164 = arith.addi %38#1, %c4 : index
      %165 = memref.load %37[%38#0, %164] : memref<?x?xf32, strided<[?, 1]>>
      %166 = vector.broadcast %165 : f32 to vector<8xf32>
      %167 = memref.load %37[%48, %164] : memref<?x?xf32, strided<[?, 1]>>
      %168 = vector.broadcast %167 : f32 to vector<8xf32>
      %169 = vector.fma %134, %166, %141 : vector<8xf32>
      %170 = memref.load %37[%52, %164] : memref<?x?xf32, strided<[?, 1]>>
      %171 = vector.broadcast %170 : f32 to vector<8xf32>
      %172 = vector.fma %134, %168, %144 : vector<8xf32>
      %173 = memref.load %37[%56, %164] : memref<?x?xf32, strided<[?, 1]>>
      %174 = vector.broadcast %173 : f32 to vector<8xf32>
      %175 = vector.fma %134, %171, %147 : vector<8xf32>
      %176 = memref.load %37[%60, %164] : memref<?x?xf32, strided<[?, 1]>>
      %177 = vector.broadcast %176 : f32 to vector<8xf32>
      %178 = vector.fma %134, %174, %150 : vector<8xf32>
      %179 = memref.load %37[%64, %164] : memref<?x?xf32, strided<[?, 1]>>
      %180 = vector.broadcast %179 : f32 to vector<8xf32>
      %181 = vector.fma %134, %177, %153 : vector<8xf32>
      %182 = memref.load %37[%68, %164] : memref<?x?xf32, strided<[?, 1]>>
      %183 = vector.broadcast %182 : f32 to vector<8xf32>
      %184 = vector.fma %134, %180, %156 : vector<8xf32>
      %185 = memref.load %37[%72, %164] : memref<?x?xf32, strided<[?, 1]>>
      %186 = vector.broadcast %185 : f32 to vector<8xf32>
      %187 = vector.fma %134, %183, %159 : vector<8xf32>
      %188 = vector.fma %134, %186, %160 : vector<8xf32>
      %189 = arith.addi %40#0, %c6 : index
      %190 = vector.load %39[%189, %40#1] : memref<?x?xf32, strided<[?, 1]>>, vector<8xf32>
      %191 = arith.addi %40#0, %c21 : index
      memref.prefetch %39[%191, %40#1], read, locality<1>, data : memref<?x?xf32, strided<[?, 1]>>
      %192 = arith.addi %38#1, %c5 : index
      %193 = memref.load %37[%38#0, %192] : memref<?x?xf32, strided<[?, 1]>>
      %194 = vector.broadcast %193 : f32 to vector<8xf32>
      %195 = memref.load %37[%48, %192] : memref<?x?xf32, strided<[?, 1]>>
      %196 = vector.broadcast %195 : f32 to vector<8xf32>
      %197 = vector.fma %162, %194, %169 : vector<8xf32>
      %198 = memref.load %37[%52, %192] : memref<?x?xf32, strided<[?, 1]>>
      %199 = vector.broadcast %198 : f32 to vector<8xf32>
      %200 = vector.fma %162, %196, %172 : vector<8xf32>
      %201 = memref.load %37[%56, %192] : memref<?x?xf32, strided<[?, 1]>>
      %202 = vector.broadcast %201 : f32 to vector<8xf32>
      %203 = vector.fma %162, %199, %175 : vector<8xf32>
      %204 = memref.load %37[%60, %192] : memref<?x?xf32, strided<[?, 1]>>
      %205 = vector.broadcast %204 : f32 to vector<8xf32>
      %206 = vector.fma %162, %202, %178 : vector<8xf32>
      %207 = memref.load %37[%64, %192] : memref<?x?xf32, strided<[?, 1]>>
      %208 = vector.broadcast %207 : f32 to vector<8xf32>
      %209 = vector.fma %162, %205, %181 : vector<8xf32>
      %210 = memref.load %37[%68, %192] : memref<?x?xf32, strided<[?, 1]>>
      %211 = vector.broadcast %210 : f32 to vector<8xf32>
      %212 = vector.fma %162, %208, %184 : vector<8xf32>
      %213 = memref.load %37[%72, %192] : memref<?x?xf32, strided<[?, 1]>>
      %214 = vector.broadcast %213 : f32 to vector<8xf32>
      %215 = vector.fma %162, %211, %187 : vector<8xf32>
      %216 = vector.fma %162, %214, %188 : vector<8xf32>
      %217 = arith.addi %40#0, %c7 : index
      %218 = vector.load %39[%217, %40#1] : memref<?x?xf32, strided<[?, 1]>>, vector<8xf32>
      %219 = arith.addi %40#0, %c22 : index
      memref.prefetch %39[%219, %40#1], read, locality<1>, data : memref<?x?xf32, strided<[?, 1]>>
      %220 = arith.addi %38#1, %c6 : index
      %221 = memref.load %37[%38#0, %220] : memref<?x?xf32, strided<[?, 1]>>
      %222 = vector.broadcast %221 : f32 to vector<8xf32>
      %223 = memref.load %37[%48, %220] : memref<?x?xf32, strided<[?, 1]>>
      %224 = vector.broadcast %223 : f32 to vector<8xf32>
      %225 = vector.fma %190, %222, %197 : vector<8xf32>
      %226 = memref.load %37[%52, %220] : memref<?x?xf32, strided<[?, 1]>>
      %227 = vector.broadcast %226 : f32 to vector<8xf32>
      %228 = vector.fma %190, %224, %200 : vector<8xf32>
      %229 = memref.load %37[%56, %220] : memref<?x?xf32, strided<[?, 1]>>
      %230 = vector.broadcast %229 : f32 to vector<8xf32>
      %231 = vector.fma %190, %227, %203 : vector<8xf32>
      %232 = memref.load %37[%60, %220] : memref<?x?xf32, strided<[?, 1]>>
      %233 = vector.broadcast %232 : f32 to vector<8xf32>
      %234 = vector.fma %190, %230, %206 : vector<8xf32>
      %235 = memref.load %37[%64, %220] : memref<?x?xf32, strided<[?, 1]>>
      %236 = vector.broadcast %235 : f32 to vector<8xf32>
      %237 = vector.fma %190, %233, %209 : vector<8xf32>
      %238 = memref.load %37[%68, %220] : memref<?x?xf32, strided<[?, 1]>>
      %239 = vector.broadcast %238 : f32 to vector<8xf32>
      %240 = vector.fma %190, %236, %212 : vector<8xf32>
      %241 = memref.load %37[%72, %220] : memref<?x?xf32, strided<[?, 1]>>
      %242 = vector.broadcast %241 : f32 to vector<8xf32>
      %243 = vector.fma %190, %239, %215 : vector<8xf32>
      %244 = vector.fma %190, %242, %216 : vector<8xf32>
      %245 = arith.addi %40#0, %c23 : index
      memref.prefetch %39[%245, %40#1], read, locality<1>, data : memref<?x?xf32, strided<[?, 1]>>
      %246 = arith.addi %38#1, %c7 : index
      %247 = memref.load %37[%38#0, %246] : memref<?x?xf32, strided<[?, 1]>>
      %248 = vector.broadcast %247 : f32 to vector<8xf32>
      %249 = memref.load %37[%48, %246] : memref<?x?xf32, strided<[?, 1]>>
      %250 = vector.broadcast %249 : f32 to vector<8xf32>
      %251 = vector.fma %218, %248, %225 : vector<8xf32>
      %252 = memref.load %37[%52, %246] : memref<?x?xf32, strided<[?, 1]>>
      %253 = vector.broadcast %252 : f32 to vector<8xf32>
      %254 = vector.fma %218, %250, %228 : vector<8xf32>
      %255 = memref.load %37[%56, %246] : memref<?x?xf32, strided<[?, 1]>>
      %256 = vector.broadcast %255 : f32 to vector<8xf32>
      %257 = vector.fma %218, %253, %231 : vector<8xf32>
      %258 = memref.load %37[%60, %246] : memref<?x?xf32, strided<[?, 1]>>
      %259 = vector.broadcast %258 : f32 to vector<8xf32>
      %260 = vector.fma %218, %256, %234 : vector<8xf32>
      %261 = memref.load %37[%64, %246] : memref<?x?xf32, strided<[?, 1]>>
      %262 = vector.broadcast %261 : f32 to vector<8xf32>
      %263 = vector.fma %218, %259, %237 : vector<8xf32>
      %264 = memref.load %37[%68, %246] : memref<?x?xf32, strided<[?, 1]>>
      %265 = vector.broadcast %264 : f32 to vector<8xf32>
      %266 = vector.fma %218, %262, %240 : vector<8xf32>
      %267 = memref.load %37[%72, %246] : memref<?x?xf32, strided<[?, 1]>>
      %268 = vector.broadcast %267 : f32 to vector<8xf32>
      %269 = vector.fma %218, %265, %243 : vector<8xf32>
      %270 = vector.fma %218, %268, %244 : vector<8xf32>
      %271 = tt.advance %arg10, [%c0_i32, %c8_i32] : <tensor<8x8xf32>>
      %272 = tt.advance %arg11, [%c8_i32, %c0_i32] : <tensor<8x8xf32>>
      scf.yield %271, %272, %251, %254, %257, %260, %263, %266, %269, %270 : !tt.ptr<tensor<8x8xf32>>, !tt.ptr<tensor<8x8xf32>>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
    }
    %25 = vector.insert %24#2, %cst_0 [0] : vector<8xf32> into vector<8x8xf32>
    %26 = vector.insert %24#3, %25 [1] : vector<8xf32> into vector<8x8xf32>
    %27 = vector.insert %24#4, %26 [2] : vector<8xf32> into vector<8x8xf32>
    %28 = vector.insert %24#5, %27 [3] : vector<8xf32> into vector<8x8xf32>
    %29 = vector.insert %24#6, %28 [4] : vector<8xf32> into vector<8x8xf32>
    %30 = vector.insert %24#7, %29 [5] : vector<8xf32> into vector<8x8xf32>
    %31 = vector.insert %24#8, %30 [6] : vector<8xf32> into vector<8x8xf32>
    %32 = vector.insert %24#9, %31 [7] : vector<8xf32> into vector<8x8xf32>
    %33 = arith.extsi %arg8 : i32 to i64
    %34 = tt.make_tensor_ptr %arg2, [%16, %20], [%33, %c1_i64], [%14, %15] {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    %35 = triton_cpu.extract_memref %34 : <tensor<8x8xf32>> -> memref<?x?xf32, strided<[?, 1]>>
    %36:2 = triton_cpu.extract_indices %34 : <tensor<8x8xf32>> -> index, index
    vector.transfer_write %32, %35[%36#0, %36#1] {in_bounds = [true, true]} : vector<8x8xf32>, memref<?x?xf32, strided<[?, 1]>>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(tt.func(triton-cpu-lower-multi-reduction),expand-strided-metadata,convert-vector-to-scf{full-unroll=true lower-scalable=false lower-tensors=false target-rank=1},lower-affine,convert-scf-to-cf,convert-index-to-llvm{index-bitwidth=0},triton-cpu-func-op-to-llvm,triton-cpu-get-program-id-op-to-llvm,triton-cpu-memory-op-to-llvm,triton-cpu-atomic-ops-to-llvm,triton-cpu-debug-ops-to-llvm,triton-cpu-math-to-vec-lib{lib=sleef},convert-math-to-llvm{approximate-log1p=true},convert-math-to-libm,convert-vector-to-llvm{enable-amx=true enable-arm-neon=false enable-arm-sve=false enable-x86vector=true force-32bit-vector-indices=true reassociate-fp-reductions=true vector-contract-lowering=outerproduct vector-transpose-lowering=eltwise},finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},reconcile-unrealized-casts,convert-arith-to-llvm{index-bitwidth=0},convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},convert-ub-to-llvm{index-bitwidth=0},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,symbol-dce,enable-line-info))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}
