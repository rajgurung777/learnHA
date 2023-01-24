To see the Help on usages type the command

```sh
python run.py --help
```



### Examples

Command to lean an HA model for switched oscillator model using two trajectories stored in the file named "data/simu_oscillator_2.txt"
    
(1) without Type Annotation
```sh
python run.py --input-filename "data/simu_oscillator_2.txt" --output-filename "oscillator_2.txt" --modes 4 --clustering-method 1 --ode-degree 1 --guard-degree 1 --segmentation-error-tol 0.100000 --threshold-correlation 0.890000 --threshold-distance 1.000000 --size-input-variable 0 --size-output-variable 2 --variable-types '' --pool-values '' --ode-speedup 50 --is-invariant 0
```
(2) with Type Annotation
```sh
python run.py --input-filename "data/simu_oscillator_2.txt" --output-filename "oscillator_2.txt" --modes 4 --clustering-method 1 --ode-degree 1 --guard-degree 1 --segmentation-error-tol 0.100000 --threshold-correlation 0.890000 --threshold-distance 1.000000 --size-input-variable 0 --size-output-variable 2 --variable-types 'x0=t1,x1=t1' --pool-values '' --ode-speedup 50 --is-invariant 0
```

