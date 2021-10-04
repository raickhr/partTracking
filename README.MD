### Particle Tracking

These codes for Langrangian Particle tracking for MOM6 oncean only Phillip's two layer baroclinic model outputs. 

The particle tracking is done for the first layer data only. 

The processes are 
1. Interpolate the variables to the same grid
2. filter the variables.
3. Make files with the required quantities like $\Pi$ and $\Lambda$ 
4. Particle track using particleTrackingFromFilteredFiles.py 
5. The files config.json is given as the runtime argument which sets the configuration for particle tracking


