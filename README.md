UCGO
----
Unstructured CFD Graph Operations.

Third Party:
------------
After git clone. Please copy the Kokkos directory into the top level.

```
git clone https://github.com/kokkos/kokkos.git
```


Compilation:
------------
For a simple CPU build:
```
mkdir build
cd build
cmake ..
make -j
```

Execution:
----------
The code can auto-generate a cartesian grid in a unit cube:
```
./build/vul/ucgo -g <num_cells_x> <num_cells_y> <num_cells_z>
```

