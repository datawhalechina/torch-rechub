"""Test HSTU imports and basic functionality."""

import sys
print("Python version:", sys.version)

try:
    print("\n测试导入...")
    
    print("1. 导入HSTULayer...")
    from torch_rechub.basic.layers import HSTULayer
    print("   ✅ HSTULayer导入成功")
    
    print("2. 导入SeqDataset和SequenceDataGenerator...")
    from torch_rechub.utils.data import SeqDataset, SequenceDataGenerator
    print("   ✅ SeqDataset和SequenceDataGenerator导入成功")
    
    print("3. 导入工具类...")
    from torch_rechub.utils.hstu_utils import RelPosBias, VocabMask, VocabMapper
    print("   ✅ RelPosBias、VocabMask、VocabMapper导入成功")
    
    print("4. 导入HSTUModel...")
    from torch_rechub.models.generative.hstu import HSTUModel
    print("   ✅ HSTUModel导入成功")
    
    print("5. 导入SeqTrainer...")
    from torch_rechub.trainers.seq_trainer import SeqTrainer
    print("   ✅ SeqTrainer导入成功")
    
    print("\n✅ 所有导入成功!")
    
    # 测试基本功能
    print("\n测试基本功能...")
    
    import torch
    import numpy as np
    
    print("1. 测试HSTULayer...")
    layer = HSTULayer(d_model=256, n_heads=8)
    x = torch.randn(2, 32, 256)
    output = layer(x)
    assert output.shape == x.shape, f"输出形状错误: {output.shape} vs {x.shape}"
    print(f"   ✅ HSTULayer正常工作，输出形状: {output.shape}")
    
    print("2. 测试HSTUModel...")
    model = HSTUModel(vocab_size=1000, d_model=256, n_heads=8, n_layers=2)
    x = torch.randint(0, 1000, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 1000), f"输出形状错误: {logits.shape}"
    print(f"   ✅ HSTUModel正常工作，输出形状: {logits.shape}")
    
    print("3. 测试SeqDataset...")
    seq_tokens = np.random.randint(0, 1000, (10, 32))
    seq_positions = np.arange(32)[np.newaxis, :].repeat(10, axis=0)
    targets = np.random.randint(0, 1000, (10,))
    dataset = SeqDataset(seq_tokens, seq_positions, targets)
    assert len(dataset) == 10, f"数据集大小错误: {len(dataset)}"
    print(f"   ✅ SeqDataset正常工作，数据集大小: {len(dataset)}")
    
    print("4. 测试SequenceDataGenerator...")
    gen = SequenceDataGenerator(seq_tokens, seq_positions, targets)
    train_loader, val_loader, test_loader = gen.generate_dataloader(batch_size=2)
    print(f"   ✅ SequenceDataGenerator正常工作")
    print(f"      - 训练集大小: {len(train_loader.dataset)}")
    print(f"      - 验证集大小: {len(val_loader.dataset)}")
    print(f"      - 测试集大小: {len(test_loader.dataset)}")
    
    print("5. 测试RelPosBias...")
    rel_pos_bias = RelPosBias(n_heads=8, max_seq_len=256)
    bias = rel_pos_bias(32)
    assert bias.shape == (1, 8, 32, 32), f"偏置形状错误: {bias.shape}"
    print(f"   ✅ RelPosBias正常工作，输出形状: {bias.shape}")
    
    print("6. 测试VocabMask...")
    mask = VocabMask(vocab_size=1000, invalid_items=[0, 1, 2])
    logits = torch.randn(2, 1000)
    masked_logits = mask.apply_mask(logits)
    assert masked_logits.shape == logits.shape, f"掩码形状错误"
    print(f"   ✅ VocabMask正常工作")
    
    print("7. 测试VocabMapper...")
    mapper = VocabMapper(vocab_size=1000)
    item_ids = np.array([10, 20, 30])
    token_ids = mapper.encode(item_ids)
    decoded_ids = mapper.decode(token_ids)
    assert np.array_equal(item_ids, decoded_ids), "映射错误"
    print(f"   ✅ VocabMapper正常工作")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

