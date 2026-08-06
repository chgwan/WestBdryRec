1. using GMagH5 time as base time. drop all GMagH5 sampling rate less than 488
2. also drop the empty amd corrput h5 shot
3. filter out GMagH5 slice,
   1. honor all slice level filter in @docs/lcfs_filters.md
4. time as a position encoding like:
   1. $$PE(pos, 2i) = \sin\left(\frac{pos}{5^{\frac{2i}{d_{\text{model}}}}}\right)$$, here we use 5 is enough. and i should be the time. 
5. generate a new npz dataset, and pls update data lineage in @docs/data_lineage
6. train all three models on the new datasets, with the same input from the previous. 
