# Hallgerd
### Deep learning framework for OpenCL

Draft of dl framework for OpenCL.
There is only Dense layer for now.

Usage:

        from hallgerd.core import Sequential
        from hallgerd.layers import Dense
        from gunnar.core import Device, Array
        
        devices = Device.getDevices()
        names = [k for k in devices]
        device = Device([devices[names[0]]])
        model = Sequential(device=device, lr=1e-3, batch_size=1024, epochs=40, loss='cross_entropy', verbose=True)
        model.add(Dense(200, 200, activation='relu'))
        model.add(Dense(200, 5, activation='softmax'))
        model.fit(X, y) 