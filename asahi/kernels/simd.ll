; SIMD-group functions, written as AIR (LLVM IR with Metal's !air.kernel), as a
; Metal compiler emits them for simd_sum(), simd_shuffle() and friends.
; out[tid * 16 + k], k = 0..15.

declare float @air.simd_shuffle.f32(float, i16)
declare i32 @air.simd_shuffle_xor.s.i32(i32, i16)
declare i32 @air.simd_shuffle_down.u.i32(i32, i16)
declare i32 @air.simd_shuffle_up.u.i32(i32, i16)
declare float @air.simd_sum.f32(float)
declare i32 @air.simd_sum.s.i32(i32)
declare i32 @air.simd_min.s.i32(i32)
declare i32 @air.simd_max.u.i32(i32)
declare i32 @air.simd_prefix_inclusive_sum.s.i32(i32)
declare i32 @air.simd_prefix_exclusive_sum.s.i32(i32)
declare float @air.simd_prefix_exclusive_sum.f32(float)
declare i64 @air.simd_ballot.i64(i1)
declare i1 @air.simd_vote_any.i64(i64)
declare i1 @air.simd_vote_all.i64(i64)
declare i32 @air.simd_broadcast.s.i32(i32, i16)
declare i32 @air.quad_shuffle_down.u.i32(i32, i16)

define void @simd_ops(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %tid, i32 %lane32) {
  %lane = trunc i32 %lane32 to i16
  %pv = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %v = load i32, ptr addrspace(1) %pv
  %f = uitofp i32 %v to float
  %base = mul i32 %tid, 16
  %o = getelementptr i32, ptr addrspace(1) %out, i32 %base
  %rev = sub i16 31, %lane
  %s = call float @air.simd_shuffle.f32(float %f, i16 %rev)
  %si = fptoui float %s to i32
  store i32 %si, ptr addrspace(1) %o
  %x = call i32 @air.simd_shuffle_xor.s.i32(i32 %v, i16 5)
  %o1 = getelementptr i32, ptr addrspace(1) %o, i32 1
  store i32 %x, ptr addrspace(1) %o1
  %fsum = call float @air.simd_sum.f32(float %f)
  %fsi = fptoui float %fsum to i32
  %o2 = getelementptr i32, ptr addrspace(1) %o, i32 2
  store i32 %fsi, ptr addrspace(1) %o2
  %isum = call i32 @air.simd_sum.s.i32(i32 %v)
  %o3 = getelementptr i32, ptr addrspace(1) %o, i32 3
  store i32 %isum, ptr addrspace(1) %o3
  %smin = call i32 @air.simd_min.s.i32(i32 %v)
  %o4 = getelementptr i32, ptr addrspace(1) %o, i32 4
  store i32 %smin, ptr addrspace(1) %o4
  %umax = call i32 @air.simd_max.u.i32(i32 %v)
  %o5 = getelementptr i32, ptr addrspace(1) %o, i32 5
  store i32 %umax, ptr addrspace(1) %o5
  %incl = call i32 @air.simd_prefix_inclusive_sum.s.i32(i32 %v)
  %o6 = getelementptr i32, ptr addrspace(1) %o, i32 6
  store i32 %incl, ptr addrspace(1) %o6
  %excl = call i32 @air.simd_prefix_exclusive_sum.s.i32(i32 %v)
  %o7 = getelementptr i32, ptr addrspace(1) %o, i32 7
  store i32 %excl, ptr addrspace(1) %o7
  %odd = and i32 %v, 1
  %c = icmp ne i32 %odd, 0
  %ballot = call i64 @air.simd_ballot.i64(i1 %c)
  %blo = trunc i64 %ballot to i32
  %o8 = getelementptr i32, ptr addrspace(1) %o, i32 8
  store i32 %blo, ptr addrspace(1) %o8
  %any = call i1 @air.simd_vote_any.i64(i64 %ballot)
  %anyi = zext i1 %any to i32
  %o9 = getelementptr i32, ptr addrspace(1) %o, i32 9
  store i32 %anyi, ptr addrspace(1) %o9
  %all = call i1 @air.simd_vote_all.i64(i64 %ballot)
  %alli = zext i1 %all to i32
  %o10 = getelementptr i32, ptr addrspace(1) %o, i32 10
  store i32 %alli, ptr addrspace(1) %o10
  %bc = call i32 @air.simd_broadcast.s.i32(i32 %v, i16 3)
  %o11 = getelementptr i32, ptr addrspace(1) %o, i32 11
  store i32 %bc, ptr addrspace(1) %o11
  %dn = call i32 @air.simd_shuffle_down.u.i32(i32 %v, i16 1)
  %o12 = getelementptr i32, ptr addrspace(1) %o, i32 12
  store i32 %dn, ptr addrspace(1) %o12
  %up = call i32 @air.simd_shuffle_up.u.i32(i32 %v, i16 2)
  %o13 = getelementptr i32, ptr addrspace(1) %o, i32 13
  store i32 %up, ptr addrspace(1) %o13
  %qd = call i32 @air.quad_shuffle_down.u.i32(i32 %v, i16 1)
  %o14 = getelementptr i32, ptr addrspace(1) %o, i32 14
  store i32 %qd, ptr addrspace(1) %o14
  %fex = call float @air.simd_prefix_exclusive_sum.f32(float %f)
  %fexi = fptoui float %fex to i32
  %o15 = getelementptr i32, ptr addrspace(1) %o, i32 15
  store i32 %fexi, ptr addrspace(1) %o15
  ret void
}

; A SIMD-group reduction kernel as Metal code writes it: simd_sum then one atomic per SIMD group.
declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

define void @simd_reduce_atomic(ptr addrspace(1) %in, ptr addrspace(1) %total, i32 %tid, i32 %lane32) {
  %pv = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %v = load i32, ptr addrspace(1) %pv
  %s = call i32 @air.simd_sum.s.i32(i32 %v)
  %first = icmp eq i32 %lane32, 0
  br i1 %first, label %add, label %done
add:
  %old = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %total, i32 %s, i32 0, i32 2, i1 true)
  br label %done
done:
  ret void
}

!air.kernel = !{!0, !5}
!0 = !{ptr @simd_ops, !{}, !{!1, !2, !3, !4}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!3 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 3, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!5 = !{ptr @simd_reduce_atomic, !{}, !{!6, !7, !3, !4}}
!6 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"total"}
