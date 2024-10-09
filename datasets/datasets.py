def create_dataset(cfg, split='train'):
    dataset = None
    data_loader = None
        
    if cfg.data.dataset == 'rcc_dataset_c': 
        from datasets.rcc_dataset_c import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            split, 
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)
        
    elif cfg.data.dataset == 'rcc_dataset_dc': 
        from datasets.rcc_dataset_dc import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            split, 
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)
        
    elif cfg.data.dataset == 'rcc_dataset_std': 
        from datasets.rcc_dataset_std import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            split, 
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)

    elif cfg.data.dataset == 'rcc_dataset_ier': 
        from datasets.rcc_dataset_ier import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            split, 
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)

    elif cfg.data.dataset == 'rcc_dataset_levir': 
        from datasets.rcc_dataset_levir import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            split, 
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)

    else:
        raise Exception('Unknown dataset: %s' % cfg.data.dataset)
    
    return dataset, data_loader
