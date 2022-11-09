# Simple EPG MRI Simulation for pytorch 
- Main function are `FSE_signal` and `MRF_Signal`
- Parameters should be broadcastable in leading batch dimensions
- All tensor inputs should work with autograd 
- Calculation will run in parallel along batch dimensions. This might cause OOM issues for large batches.
