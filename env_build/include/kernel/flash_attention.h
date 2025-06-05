void flash_attention_fwd_cpp(
    const float* __restrict q_ptr,
    const float* __restrict k_ptr,
    const float* __restrict v_ptr,
    float* __restrict out_ptr,
    float* __restrict m_logsumexp_ptr, // Stores m_i + log(l_i)

    int Z, int H, int N_CTX_Q, int N_CTX_K, int D_HEAD,
    float sm_scale,
    int BLOCK_M, int BLOCK_N, // BLOCK_DMODEL is implicitly D_HEAD
    bool is_causal
);