# script to pull Adrian Sheppard's voxelized files
# accessible using gzip, np.fromstring, and 'int8' as dtype
f1=http://people.physics.anu.edu.au/~aps110/network_comparison/REFERENCE_CASES/BEADPACK/segmented_bead_pack_512.ubc.gz
f2=http://people.physics.anu.edu.au/~aps110/network_comparison/REFERENCE_CASES/CASTLE/segmented_castle_512.ubc.gz
f3=http://people.physics.anu.edu.au/~aps110/network_comparison/REFERENCE_CASES/LRC32/segmented_lrc32_512.ubc.gz
f4=http://people.physics.anu.edu.au/~aps110/network_comparison/REFERENCE_CASES/GAMBIER/segmented_Gambier_512.ubc.gz

for f in $f1 $f2 $f3 $f4
do
    curl -O $f
done
