# Riemannian Diffusion


## Protein backbone dihedral angle dataset

To visualize the rama scatter plot

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/top500/aggregated_angles.tsv', 
                   delimiter='\t', 
                   names=['source', 'phi', 'psi', 'amino'])

fig = plt.figure()
for j, amino in enumerate(['General', 'Glycine', 'Proline', 'Pre-Pro']):
    x = data[data['amino'] == amino][['phi', 'psi']].values
    ax = fig.add_subplot(2, 2, j+1)
    plt.scatter(x[:, 0], x[:, 1], s=1, alpha=0.01)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.title(amino)

plt.tight_layout()
```

It should produce the following fig.
![alt text](assets/rama_scatter.png)


### Dataset size

| Tori  | General | Glycine | Proline | Pre-Pro | RNA |
| :----: | :----:  | :----:  | :----:  | :----:  | :----: |
| Train | 110566 | 10626 | 6107 | 5528 | 7582 |
| Valid | 13821 | 1328 | 763 | 691 | 948 |
| Test | 13821 | 1329 | 764 | 691 | 948 |
| All | 138208 | 13283 | 7634 | 6910 | 9478  |
| Dim | 2 | 2 | 2 | 2 | 7 |
