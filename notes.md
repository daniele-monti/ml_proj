## Duplicates in dataset
According to the research paper by the original creators of the dataset:
> The data were recorded by a computerized system (iLab), which automatically manages the
> process of wine sample testing from producer requests to laboratory and sensory
> analysis. Each entry denotes a given test (analytical or sensory) [...]

They also say:
> During the preprocessing stage, the database was transformed in order to
> include a distinct wine sample (with all tests) per row.

which leads me to the conclusion that rows with the exact same feature values corresponds to different taste tests rather than to different wines samples that happen to have the same feature values.
If this is the case, it is surprising that in all of such cases, each taste test resulted in the same quality evaluation from the experts, but since the original authors decided to use the dataset without dropping the duplicated rows, I will do the same.
