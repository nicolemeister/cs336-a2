import torch
import triton
from einops import rearrange
import math

'''
Your implementation should take input Q, K, and V as well as a flag is_causal and produce
the output O and the logsumexp value L. You can ignore the is_causal flag for this task. The
autograd.Function forward should then save L, Q, K, V for the backward pass and return
O. Remember that the implementation of the forward method of autograd.Function always
takes the context as its first parameter. Any autograd.Function class needs to implement a
backward method, but for now you can make it just raise NotImplementedError. If you need
something to compare against, you can implement Equation 4 to 6 and 12 in PyTorch and
compare your outputs.
The interface is then def forward(ctx, Q, K, V, is\_causal=False). Determine your own
tile sizes, but make sure they are at least of size 16 × 16. We will always test your code with
dimensions that are clean powers of 2 and at least 16, so you don't need to worry about
out-of-bounds accesses.

'''


class FlashAttention2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get dimensions
        batch_size = Q.shape[0]
        N_q, N_k = Q.shape[1], K.shape[1]  # sequence lengths
        d = Q.shape[-1]  # embedding dimension
        
        # Set tile sizes
        B_q, B_k = 16, 16  # Tile sizes of at least 16x16 as required
        
        # Calculate number of tiles
        T_q = (N_q + B_q - 1) // B_q  # Ceiling division
        T_k = (N_k + B_k - 1) // B_k
        
        # Initialize output tensors
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, N_q, device=Q.device)
        
        # Process each batch
        for b in range(batch_size):
            # Create tiles for this batch
            Q_tiles = [Q[b, i*B_q:min((i+1)*B_q, N_q)] for i in range(T_q)]
            K_tiles = [K[b, j*B_k:min((j+1)*B_k, N_k)] for j in range(T_k)]
            V_tiles = [V[b, j*B_k:min((j+1)*B_k, N_k)] for j in range(T_k)]
            
            for i in range(T_q):
                # Get current Q tile
                Q_i = Q_tiles[i]  # [B_q, d]
                current_B_q = Q_i.shape[0]  # Handle last tile which might be smaller

                # Initialize for this tile
                O_i = torch.zeros(current_B_q, d, device=Q.device)
                l_i = torch.zeros(current_B_q, device=Q.device)
                m_i = torch.full((current_B_q,), -float('inf'), device=Q.device)

                for j in range(T_k):
                    K_j = K_tiles[j]  # [B_k, d]
                    V_j = V_tiles[j]  # [B_k, d]
                    current_B_k = K_j.shape[0]  # Handle last tile which might be smaller

                    # Compute attention scores for this tile using PyTorch matmul
                    S_ij = torch.matmul(Q_i, K_j.T) / torch.sqrt(torch.tensor(d, device=Q.device))  # [B_q, B_k]

                    # Update running max
                    m_i_new = torch.maximum(m_i, torch.max(S_ij, dim=1).values)

                    # Compute P_ij
                    P_ij = torch.exp(S_ij - m_i_new.unsqueeze(1))  # [B_q, B_k]

                    # Update l_i
                    l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.sum(P_ij, dim=1)

                    # Update O_i using PyTorch matmul
                    O_i_new = torch.matmul(torch.diag_embed(torch.exp(m_i - m_i_new)), O_i) + torch.matmul(P_ij, V_j)

                    # Update for next iteration
                    m_i = m_i_new
                    l_i = l_i_new
                    O_i = O_i_new

                # Final computation for this tile using PyTorch matmul
                O_i = torch.matmul(torch.diag_embed(1.0 / l_i), O_i)
                L_i = m_i + torch.log(l_i)

                # Store results in the correct slice of O and L
                start_idx = i * B_q
                end_idx = min((i + 1) * B_q, N_q)
                O[b, start_idx:end_idx] = O_i[:end_idx-start_idx]
                L[b, start_idx:end_idx] = L_i[:end_idx-start_idx]

        ctx.save_for_backward(L, Q, K, V)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Implement the backward pass for your FlashAttention-2 autograd.Function using PyTorch (not
        Triton) and torch.compile. Your implementation should take the Q, K, V, O, dO, L and tensors as
        output, and return dQ, dK and dV. Remember to compute and use the D vector.
        '''
        L, Q, K, V = ctx.saved_tensors
        dO = grad_output
        d = Q.shape[-1]  # embedding dimension
        
        # Set tile sizes
        B_q, B_k = 16, 16  # Tile sizes of at least 16x16 as required
        
        scale = 1 / torch.sqrt(torch.tensor(d, device=Q.device))

         # Compute S = QK^T/√d
        # Use transpose(-2, -1) instead of .T to only transpose last two dimensions
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [batch, N_q, N_k]
        
        # Compute P = exp(S - L.unsqueeze(-1))
        P = torch.exp(S - L.unsqueeze(-1))  # [batch, N_q, N_k]
        
        # Compute D = sum(P ◦ (dO V^T), dim=2)
        dOV = torch.matmul(grad_output, V.transpose(-2, -1))  # [batch, N_q, N_k]
        D = torch.sum(P * dOV, dim=2)  # [batch, N_q]
        
        # Compute gradients
        # dV = P^T dO
        dV = torch.matmul(P.transpose(-2, -1), grad_output)  # [batch, N_k, d]
        
        # dS = P ◦ (dOV^T - D.unsqueeze(-1))
        dS = P * (dOV - D.unsqueeze(-1))  # [batch, N_q, N_k]
        
        # dQ = dS K/√d
        dQ = torch.matmul(dS, K) * scale  # [batch, N_q, d]
        
        # dK = dS^T Q/√d
        dK = torch.matmul(dS.transpose(-2, -1), Q) * scale  # [batch, N_k, d]

        return dQ, dK, dV, None