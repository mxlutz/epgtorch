# Simple EPG for pytorch
- Main function is `FSE_signal`
- Parameters can be should be broadcastable in leading batch dimensions
- All tensor inputs should work with autograd 
- Calculation will run in parallel along batch dimensions. This might cause OOM issues for large batches.