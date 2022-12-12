import cv2
import numpy as np
import math


def main():
  trainingImagesPath = [
    ".\dataset\Training\subject01.happy.jpg",
    ".\dataset\Training\subject02.normal.jpg",
    ".\dataset\Training\subject03.normal.jpg",
    ".\dataset\Training\subject07.centerlight.jpg",
    ".\dataset\Training\subject10.normal.jpg",
    ".\dataset\Training\subject11.normal.jpg",
    ".\dataset\Training\subject14.normal.jpg",
    ".\dataset\Training\subject15.normal.jpg",
  ]
  testingImagesPath = [
    ".\dataset\Testing\subject01.normal.jpg",
    ".\dataset\Testing\subject07.happy.jpg",
    ".\dataset\Testing\subject07.normal.jpg",
    ".\dataset\Testing\subject11.happy.jpg",
    ".\dataset\Testing\subject14.happy.jpg",
    ".\dataset\Testing\subject14.sad.jpg",
  ]
  imageRow = 231
  imageColumn = 195

  # For each training image, stack the row together to form a column vector R
  columnVectors = get_column_vector_set(trainingImagesPath)
  # The mean face is computed by taking the average of the training face images
  meanFace = get_mean_face(columnVectors)
  show_mean_face(meanFace, imageRow, imageColumn)
  # We subtract the mean face from each training face and put them into a single matrix A
  # The follow function return A transpose
  AT = get_normalized_column_vector_set(columnVectors, meanFace)
  A = AT.transpose()
  # We find eigenvalues of L = A transpose x A
  L = np.matmul(AT, A)
  # Put eigenvectors of L into a single matrix V
  _, V = np.linalg.eig(L)
  # The M largest eigenvectors of C can be found by U = AV
  U = np.matmul(A,V)
  UT = U.transpose()
  # Display eigenface
  show_eigenface(UT, imageRow, imageColumn)
  # For each training face, calculate its eigenface coefficients omega
  trainingCoefficients = []
  for R in AT:
    omega = np.matmul(UT, R)
    trainingCoefficients.append(omega)

  # Calculate the eigenface coefficient of each testing image
  testingColumnVectors = get_column_vector_set(testingImagesPath)
  testingNormalizedColumnVectors = get_normalized_column_vector_set(testingColumnVectors, meanFace)
  testingCoefficients = []
  for I in testingNormalizedColumnVectors:
    omega = np.matmul(UT, I)
    testingCoefficients.append(omega)

  # Calculate and print recognition result for each test image
  for i in range(len(testingCoefficients)):
    print(testingImagesPath[i] + " is match to ")
    matchIndex = get_match_face(testingCoefficients[i], trainingCoefficients)
    print(trainingImagesPath[matchIndex])
    print()



def get_training_images(trainingImagesPath):
  images_set = []
  for path in trainingImagesPath:
    # read image in grayscale mode
    image = cv2.imread(path, 0)
    images_set.append(image)
  return images_set


def get_column_vector(image):
  #convert n * n image to n^2 * 1 column vector
  columnVector = [0]*(len(image)*len(image[0]))
  index = 0
  for row in image:
    for column in row:
      columnVector[index] = column
      index += 1
  return columnVector


def get_column_vector_set(trainingImagesPath):
  # convert each training images to a column vector R
  image_set = get_training_images(trainingImagesPath)
  vector_set = []
  for image in image_set:
    vector_set.append(get_column_vector(image))
  return vector_set


def get_mean_face(columnVectors):
  # calculate average face out of all column vectors
  meanFace = [0]*len(columnVectors[0])
  for cv in columnVectors:
    for i in range(len(cv)):
      meanFace[i] += cv[i]
  for i in range(len(meanFace)):
    # round off digit after decimal points because pixels only accept int as legal value
    meanFace[i] = round(meanFace[i]/len(columnVectors))
  return np.array(meanFace)


def get_normalized_column_vector_set(columnVectors, meanFace):
  # normalize column vectors by subtracting the average face from them
  columnVectors = np.array(columnVectors)
  for i in range(len(columnVectors)):
    columnVectors[i] = columnVectors[i] - meanFace
  return columnVectors


def get_euclidean_distance(a, b):
  # calculate euclidean distance between two vectors
  dist = 0
  for i in range(len(a)):
    dist += (a[i] - b[i])**2
  return math.sqrt(dist)


def get_match_face(input, dataset):
  # find the face in dataset with minimal euclidean distance from the input
  matchIndex = 0
  minDistance = float('inf')
  for i in range(len(dataset)):
    curDistance = get_euclidean_distance(input,dataset[i])
    if(curDistance < minDistance):
      minDistance = curDistance
      matchIndex = i
  return matchIndex


def show_image(image, name = "image"):
    # open the image in a new window
    # press 0 key to close images
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_mean_face(meanFace, imageRow, imageColumn):
  # the dimension of meanFace is N^2 x 1
  print("Displaying mean face")
  print("Press 0 to close the image")
  print()
  image = np.uint8(np.reshape(meanFace, (imageRow, imageColumn)))
  show_image(image)


def show_eigenface(UT, imageRow, imageColumn):
  i = 1
  for eigenface in UT:
    print("Displaying eigenface ", i)
    i += 1
    print("Press 0 to close the images")
    print()
    image = np.uint8(np.reshape(eigenface, (imageRow, imageColumn)))
    show_image(image)


main()