# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder3D',
    img_shape=(40, 256, 256),   # Input volume dimensions (D, H, W) 
    input_dim=1,                # Number of input channels
    output_dim=2,               # Number of segmentation classes
    embed_dim=768,              # Transformer embedding dimension
    patch_size=(8,16,16),              # Patch size for vision transformer
    num_heads=12,               # Number of attention heads
    dropout=0.1,                # Dropout rate
    train_cfg=dict(),
    test_cfg=dict(),
    decode_head=dict(
        num_classes=2
    )
)