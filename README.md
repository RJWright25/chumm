# Code for Halo accUMulation of Mass [CHUMM]
## AUTHOR: Ruby Wright (2019)
#### PhD Student | International Centre for Radio Astronomy Research (ICRAR) | University of Western Australia (UWA)

![](chumm.jpeg)


Tools to calculate accretion rates to halos in cosmological galaxy formation simulations.

## USAGE

### Step 0: Requirements for usage

To run CHUMM over a cosmological simulation, VELOCIraptor and TreeFrog must first be run on the simulation snapshots. 
Create a directory with an informative title where you would like the accretion data outputs.

### Step 1: Generate halo data

First, edit the GenData-HaloData.py script (with command line arguments) to generate lists (for each snap) pointing towards:

(1) Particle data

(2) VELOCIraptor data

(3) TreeFrog data

These lists must have identical length. If some snapshots are not available, they must be padded in these lists with 'None'.

CHUMM collates all of the halo data from VELOCIraptor and TreeFrog data into a list (for each snap) of dictionaries with 
keys for relevant halo properties. This object is saved to file in pickle/binary form in 3 levels of increasing detail. Each
of these also contains pointers to the particle/VELOCIraptor/TreeFrog data which is used upstream.

The 'B1' file contains the minimal basic halo data required for accretion calculation (saves memory at runtime).

The 'B2' file contains the minimal basic halo data required for accretion calculation, as well as detailed TreeFrog data.

The 'B3' file contains all available halo data from VELOCIraptor and TreeFrog, to be used for data processing. 

### Step 2: Generate particle histories

Secondly, use the GenData-PartData.py (with command line arguments) to generate particle histories.

In order to calculate accretion rates of various types, CHUMM must track the histories of particles across simulation time
to determine which particles have been processed in a halo environment, and which are primordial. This is done in a 2-step manner,
first sorting and indexing particles and recording their host structure (from VELOCIraptor, of -1 if not part of structure) which 
can be done for each snap individually. Subsequently, these histories are integrated in serial over each snapshot to add a counter 
for each particle indicating how many snaps in the past (including the current snap) the particle has been part of structure. 

CHUMM saves this data in the form of a hdf5 file, with a group for each particle type datasets outlined in AccretionTools.py.

### Step 3: Generate accretion data

Thirdly, use the gen_accretion_data_fof_serial function (with command line arguments) to generate accretion data for a given snap. 

CHUMM calculates accretion (outflow) rates to halos identified in VELOCIRAPTOR by considering which particles have entered (left) an object
at the current snap compared to a snap at a user-specified depth (the interval being snap_1 to snap_2, where snap_2-snap_1=pre_depth). The fate of a particle at a subsequent snapshot is also tracked at a user-specified depth (snap_3=snap_2+post_depth). This calculation is parallelised with halo-based splitting, with the algorithm independent for each halo. 

CHUMM saves this data in the form of a hdf5 file, with a group for each halo, grouped by inflow/outflow, further grouped by particle type,
saving particle IDs, masses, a flag indicating whether the particle stayed (returned) in the halo at the 3rd snapshot ('fidelity'), and the
particle's previous (new) host. Any gas particle data can be added to the accretion outputs with add_gas_particle_data. There is a hdf5 file
for each parallel process spawned, based on the number of processes specified by the user.

The function postprocess_acc_data_serial processes the raw accretion outputs above and sums particle fluxes and masses in various
combinations, pertaining to different classifications of accretion events. This data is saved for all halos to a single summed output file,
forming a database of integrated accretion data ordered by halo index (without detailed data for every outflow and inflow particle).
