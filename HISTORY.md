# History
# 0.1.2 (2023-12-07)

- added kwarg min_q111 for function fit_structurefactor()
- added return of r_squared in fit_formfactor()
- log filename changed to 'nanoscatterfit.log'
- delete_log_file() doesnt remove the file anymore but just erases its content in order to prevent it from not being recreated in a running jupyter notebook file
- function isscatter() now also checks if the 111 peak intensity is unexpectedly small


# 0.1.1 (2023-11-25)

- fixed issues with detecting irrelevant peaks as the (111) peak in the function plot_structuremodel()
- added examples for batch analyzing suspensions and superstructures

# 0.1.0 (2023-11-15)

- First release on GitHub.


