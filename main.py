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
  meanFace = np.mean(columnVectors, axis=0)
  show_mean_face(meanFace, imageRow, imageColumn)
  # We subtract the mean face from each training face and put them into a single matrix A
  # The follow function return A transpose
  A = get_normalized_column_vector_set(columnVectors, meanFace)
  AT = A.transpose()
  # We find eigenvalues of L = A transpose x A
  L = np.matmul(AT, A)
  # Put eigenvectors of L into a single matrix V
  _, V = np.linalg.eig(L)
  # The M largest eigenvectors of C can be found by U = AV
  U = np.matmul(A,V)
  U = normalize_U(U)
  UT = U.transpose()
  # Display eigenface
  show_eigenface(UT, imageRow, imageColumn)
  # For each training face, calculate its eigenface coefficients omega
  trainingCoefficients = np.zeros(shape=(len(trainingImagesPath),len(trainingImagesPath)))
  for i in range(len(trainingImagesPath)):
    omega = np.matmul(UT, A[:,i])
    trainingCoefficients[i]=omega
  # Calculate the eigenface coefficient of each testing image
  testingColumnVectors = get_column_vector_set(testingImagesPath)
  testingNormalizedColumnVectors = get_normalized_column_vector_set(testingColumnVectors, meanFace)
  testingCoefficients = np.zeros(shape=(len(testingImagesPath),len(trainingImagesPath)))
  for i in range(len(testingImagesPath)):
    omega = np.matmul(UT, testingNormalizedColumnVectors[:,i])
    print(omega)
    testingCoefficients[i]=omega

  # Calculate and print recognition result for each test image
  for i in range(len(testingCoefficients)):
    print(testingImagesPath[i] + " is match to ")
    matchIndex = get_match_face(testingCoefficients[i], trainingCoefficients)
    print(trainingImagesPath[matchIndex])
    print()


def get_column_vector_set(trainingImagesPath):
  #flat out images to columnvectors
  column_vector_set = 0
  for i in range(len(trainingImagesPath)):
    # read image in grayscale mode
    image = cv2.imread(trainingImagesPath[i], 0)
    image = image.reshape(1,45045)
    if i == 0:
      column_vector_set = image
    else:
      column_vector_set = np.append(column_vector_set,image,axis=0)
  return column_vector_set

def get_normalized_column_vector_set(columnVectors, meanFace):
  # normalize column vectors by subtracting the average face from them
  normalize_cv = np.ndarray(shape=(len(columnVectors),45045))
  for i in range(len(columnVectors)):
    normalize_cv[i] = columnVectors[i] - meanFace
  return normalize_cv.T

def normalize_U(U):
  # normalize to 0-255
  Umin=np.amin(U)
  Umax=np.amax(U)
  U2=np.ndarray(shape=(45045,8))
  for i in range(U.shape[0]):
    for j in range(U.shape[1]):
      g = U[i][j]
      g = (g - Umin) * 255/(Umax - Umin)
      U2[i][j]=g
  return U2

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
    curDistance = get_euclidean_distance(input,dataset[i].reshape(8,1))
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