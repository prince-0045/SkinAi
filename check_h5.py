import h5py
path='backend/app/services/effnetv2s_final.weights.h5'
with h5py.File(path,'r') as f:
    print('top keys', list(f.keys()))
    layers=f['layers'] if 'layers' in f else None
    if layers is not None:
        names=list(layers.keys())
        print('layers count', len(names))
        print('first 20', names[:20])
        print('last 20', names[-20:])
        for n in ['dense','dense_1','dense_2']:
            if n in layers:
                print('\n===', n, '===')
                g = layers[n]
                if 'vars' in g:
                    vars_group = g['vars']
                    for var_name in ['0','1']:
                        if var_name in vars_group:
                            d = vars_group[var_name]
                            print('var', var_name, 'shape', d.shape, 'dtype', d.dtype)
                            # print name attribute if available
                            try:
                                print('name attr', d.attrs.get('name'))
                            except Exception:
                                pass

