import SimpleITK as sitk
import tifffile as tif

volume = tif.imread("final.tif")
print(volume.shape, volume.dtype)

# Convert the numpy array to a SimpleITK image
sitk_image = sitk.GetImageFromArray(volume)

# Apply a binary threshold to create a binary image
binary_image = sitk.BinaryThreshold(
    sitk_image, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0
)

# Perform connected component analysis
cc = sitk.ConnectedComponent(binary_image)

# Use LabelShapeStatisticsImageFilter to count the number of labels
label_stats = sitk.LabelShapeStatisticsImageFilter()
label_stats.Execute(cc)

# Get the number of labels (cells)
num_cells = label_stats.GetNumberOfLabels()

# Optionally, convert the labeled image back to a numpy array to inspect the result
labeled_volume = sitk.GetArrayFromImage(cc)

print("Number of cells:", num_cells)
# print("Labeled volume:\n", labeled_volume)
