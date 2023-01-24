import grid as gridSearch


# ******************************************

rate, param = gridSearch.find_parameters('heart_scale', '-log2c -1,1,1 -log2g -1,1,1')
print("param is ", param)

# ******************************************

