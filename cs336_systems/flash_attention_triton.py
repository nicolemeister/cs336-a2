import triton
import triton.language as tl
import torch

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    key_tile_index = 0 
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
        # shape=(N_QUERIES, 1),
        # strides=(stride_lq, 1),
        # offsets=(query_tile_index * Q_TILE_SIZE, 0),
        # block_shape=(Q_TILE_SIZE, 1),
        # order=(1, 0)
    )
    
    # Initialize a buffer to write to
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32) # [Q_TILE_SIZE, D]

    # Convert scale from pointer to value
    scale = tl.load(scale)

    # Create query indices for causal masking
    # get a range of tile indices from 0-->Q_TILE_SIZE, then shift it based on the query_tile_index * tile size 
    query_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE

    # loop through the key tiles 
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32) # [B_k, d]
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32) # [B_k, d]

        # Create key indices for causal masking
        # get a range of tile indices from 0-->K_TILE_SIZE, then shift it based on the key_tile_index
        key_indices = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE

        # Compute attention scores
        S_j = tl.dot(Q_i, tl.trans(K_j)) * scale

        # Apply causal mask if needed
        if is_causal:
            # Create mask where query_indices >= key_indices for causal attention
            # We need to reshape for proper broadcasting
            causal_mask = query_indices[:, None] >= key_indices[None, :]
            # Apply mask: keep values where mask is True, replace with -1e6 where mask is False
            S_j = tl.where(causal_mask, S_j, -1e6)

        # update new max: tl.max for reduction, tl.maximum for element-wise comparison
        m_new = tl.maximum(m, tl.max(S_j, axis=1))
        
        # Compute P_ij - reshape m_new to broadcast correctly
        P_j = tl.exp(S_j - m_new[:, None])  # [B_q, B_k] # Use None/newaxis for broadcasting

        # Update l_i        
        l_new = tl.exp(m - m_new) * l + tl.sum(P_j, axis=1)
        
        # Cast P˜(j)i to the dtype of V(j) before multiplying them
        P_j = P_j.to(V_j.dtype)
        # Create diagonal matrix for exp(m - m_new)
        exp_diff = tl.exp(m - m_new)
        # Scale O directly without creating full diagonal matrix
        # O_scaled = O * exp_diff[:, None]  # Broadcasting for row-wise scaling
        # O_new = tl.dot(P_j, V_j, acc=O_scaled)

        O_new = tl.dot(P_j, V_j, acc=O * exp_diff[:, None])

        # Update for next iteration
        m = m_new
        l = l_new
        O = O_new

        # move pointers to the next tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
    # cast Oi to the appropriate dtype before writing it to global memory
    # output = O.to(O.dtype)

    # *_block_ptr.type.element_type
    # output_block_ptr.type.element_type
    # Normalize each row of O by dividing by the corresponding l value
    O = (1.0/l[:, None]) * O 

    l = m + tl.log(l)

    # Write output to the output block pointer (a single scalar per row).    
    tl.store(O_block_ptr, O.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l.to(L_block_ptr.type.element_ty), boundary_check=(0,))


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # set the tile sizes 
        # B_q, B_k = 16, 16
        B_q = min(64, triton.next_power_of_2(Q.shape[1]))  # Adjust tile size to be compatible
        B_k = min(64, triton.next_power_of_2(K.shape[1]))  # and not too large
        
        batch_size = Q.shape[0]
        N_q, N_k = Q.shape[1], K.shape[1]  # sequence lengths
        d = Q.shape[-1]  # embedding dimension

        # Calculate number of tiles
        T_q = (N_q + B_q - 1) // B_q  # Ceiling division
        T_k = (N_k + B_k - 1) // B_k

        ctx.D = d
        ctx.Q_TILE_SIZE = B_q
        ctx.K_TILE_SIZE = B_k
        
        # Initialize final output tensors
        O = torch.empty(batch_size, N_q, d, device=Q.device)
        L = torch.empty(batch_size, N_q, device=Q.device)

        # launch the kernel with size Tq, batch_size
        flash_fwd_kernel[(T_q, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_q, N_KEYS=N_k,
            scale = 1/torch.sqrt(torch.tensor(d, device=Q.device)),
            D=ctx.D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal
        )
        ctx.save_for_backward(L)
        ctx.is_causal = is_causal
            
        return O
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")