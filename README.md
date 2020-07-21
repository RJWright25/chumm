# Code for Halo accUMulation of Mass [CHUMM]
#### RUBY WRIGHT (PhD Student | ICRAR-UWA) (2019-)

![](Logos/chumm_jaws.jpeg)

Tools to calculate accretion rates to halos in cosmological galaxy formation simulations.

## USAGE

### Step 0: VELOCIraptor/TreeFrog outputs from cosmological simulation

To run CHUMM over a cosmological simulation, VELOCIraptor and TreeFrog must first be run on the simulation snapshots. 
Create a directory with an informative title where you would like the accretion data outputs.

### Step 1: Generate halo data

First, edit the GenData-HaloData.py and GenData-HaloData-run.py scripts to generate lists (for each snap) pointing towards:

(1) Particle data

(2) VELOCIraptor data

(3) TreeFrog data

These lists must have identical length. If some snapshots are not available, they must be padded in these lists with 'None'.

CHUMM collates all of the halo data from VELOCIraptor and TreeFrog data into a list (for each snap) of dictionaries with 
keys for relevant halo properties. This object is saved to file in pickle/binary form in 3 levels of increasing detail. Each
of these also contains pointers to the particle/VELOCIraptor/TreeFrog data which is used upstream.

* The 'B1' file contains the minimal basic halo data required for accretion calculation (saves memory at runtime).
* The 'B2' file contains the minimal basic halo data required for accretion calculation, with detailed TreeFrog data.
* The 'B3' file contains all available halo data from VELOCIraptor and TreeFrog, to be used for data processing. 
* The 'B4' file contains a specified subset of all available halo data from VELOCIraptor and TreeFrog, to be used for data processing. 

### Step 2: Generate particle histories

Secondly, use the GenData-PartData-run.py script (with command line arguments) to generate particle histories.

In order to calculate accretion rates of various types, CHUMM must track the histories of particles across simulation time
to determine which particles have been processed in a halo environment, and which are primordial. 

CHUMM saves this data in the form of a hdf5 file, with a group for each particle type datasets outlined in ParticleTools.py.

### Step 3: Generate accretion data

Thirdly, use the GenData-AccData-run.py script to generate accretion data for a given snap. All outputs are in hdf5 format. 
Particle data can be output for each accretion candidate particle with the write_partdata flag, which is needed for further post-processing.
Otherwise, only integrated accretion rates are saved. 
 
CHUMM saves accretion data in the form of a hdf5 file, with specific data structures and algorithms outlined in AccretionTools.py.

### Step 4: Post-processing 

The user can further post-process accretion particle data to:

1. Track back and save recycled particles to their last structure (GenData-Recycling-run.py)
2. Add additional particle properties from raw particle data to accretion outputs (GenData-Properties-run.py)
3. Average and save accreted particle properties (requires 1. and 2., GenData-AveProps-run.py)
