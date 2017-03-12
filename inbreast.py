#import dicom # some machines not install pydicom
import scipy.misc
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import cPickle
#import matplotlib
#import matplotlib.pyplot as plt 
from skimage.filters import threshold_otsu
import os
from os.path import join as join
import csv
import scipy.ndimage
import dicom
#import cv2
path = '../AllDICOMs/'
preprocesspath = '../preprocesspath/'
labelfile = './label.txt'

def readlabel():
  '''read the label as a dict from labelfile'''
  mydict = {}
  with open(labelfile, 'r') as f:
    flines = f.readlines()
    for line in flines:
      data = line.split()
      if int(data[1]) == 0:
        mydict[data[0]] = int(data[1])
      else:
        assert(int(data[1])==2 or int(data[1])==1)
        mydict[data[0]] = int(data[1])-1
  return mydict

def readdicom(mydict):
  '''read the dicom image, rename it consistently with the name in labels, crop and resize, and save as pickle.
  mydict is the returned value of readlabel'''
  img_ext = '.dcm'
  img_fnames = [x for x in os.listdir(path) if x.endswith(img_ext)]
  for f in img_fnames:
    names = f.split('_')
    if names[0] not in mydict:
      print(names[0]+'occur error')
    dicom_content = dicom.read_file(join(path,f))
    img = dicom_content.pixel_array
    '''fig = plt.figure()
    ax1 = plt.subplot(3,3,1)
    ax2 = plt.subplot(3,3,2)
    ax3 = plt.subplot(3,3,3)
    ax4 = plt.subplot(3,3,4)
    ax5 = plt.subplot(3,3,5)
    ax6 = plt.subplot(3,3,6)
    ax7 = plt.subplot(3,3,7)
    ax8 = plt.subplot(3,3,8)
    ax9 = plt.subplot(3,3,9)
    ax1.imshow(img, cmap='Greys_r')
    ax1.set_title('Original')
    ax1.axis('off')'''
    
    thresh = threshold_otsu(img)
    binary = img > thresh
    #ax2.imshow(binary, cmap='Greys_r')
    #ax2.set_title('mask')
    #ax2.axis('off')
    
    minx, miny = 0, 0
    maxx, maxy = img.shape[0], img.shape[1]
    for xx in xrange(img.shape[1]):
      if sum(binary[xx, :]==0) < binary.shape[1]-60:
        minx = xx
        break
    for xx in xrange(img.shape[0]-1,0,-1):
      if sum(binary[xx, :]==0) < binary.shape[1]-60:
        maxx = xx
        break
    if names[3] == 'R':
      maxy = img.shape[1]
      for yy in xrange(int(img.shape[1]*3.0/4), -1, -1):
        if sum(binary[:,yy]==0) > binary.shape[0]-10: 
          miny = yy
          break
    else:
      miny = 0
      for yy in xrange(int(img.shape[1]/4.0), img.shape[1], 1):
        if sum(binary[:,yy]==0) > binary.shape[0]-10: 
          maxy = yy
          break
    print(minx, maxx, miny, maxy)
    #ax3.set_title('Foreground')
    #ax3.imshow(img[minx:maxx+1, miny:maxy+1], cmap='Greys_r')
    #ax3.axis('off')
    
    img = img.astype(np.float32)
    img1 = scipy.misc.imresize(img[minx:maxx+1, miny:maxy+1], (227, 227), interp='cubic')
    with open(join(preprocesspath, names[0])+'227.pickle', 'wb') as outfile:
      cPickle.dump(img1, outfile) 
    img1 = scipy.misc.imresize(img[minx:maxx+1, miny:maxy+1], (299, 299), interp='cubic')
    with open(join(preprocesspath, names[0])+'299.pickle', 'wb') as outfile:
      cPickle.dump(img1, outfile) 
    '''ax4.set_title('Resize')
    ax4.imshow(img, cmap='Greys_r')
    ax4.axis('off')

    img = img.astype(np.float32)
    img -= np.mean(img)
    img /= np.std(img)
    ax5.set_title('Norm')
    ax5.imshow(img, cmap='Greys_r')
    ax5.axis('off')
    with open(join(preprocesspath, names[0])+'norm.pickle', 'wb') as outfile:
      cPickle.dump(img, outfile)
      #imgshape = img.shape
    
    img = np.fliplr(img)
    ax6.set_title('Flip')
    ax6.imshow(img, cmap='Greys_r')
    ax6.axis('off')
    
    num_rot = np.random.choice(4)               #rotate 90 randomly
    img = np.rot90(img, num_rot)
    ax7.set_title('Rotation')
    ax7.imshow(img, cmap='Greys_r')
    ax7.axis('off')
    fig.savefig(join(preprocesspath, names[0])+'.jpg')
    plt.close(fig)'''

def cvsplit(fold, totalfold, mydict):
  '''get the split of train and test
  fold is the returned fold th data, from 0 to totalfold-1
  total fold is for the cross validation
  mydict is the return dict from readlabel'''
  skf = StratifiedKFold(n_splits=totalfold)  # default shuffle is false, okay!
  #readdicom(mydict)
  y = mydict.values()
  x = mydict.keys()
  count = 0
  for train, test in skf.split(x,y):
    print(len(train), len(test))
    if count == fold:
      #print test
      return train, test
    count += 1

def cvsplitenhance(fold, totalfold, mydict, valfold=-1):
  '''get the split of train and test
  fold is the returned fold th data, from 0 to totalfold-1
  total fold is for the cross validation
  mydict is the return dict from readlabel
  sperate the data into train, validation, test'''
  skf = StratifiedKFold(n_splits=totalfold)  # default shuffle is false, okay!
  #readdicom(mydict)
  y = mydict.values()
  x = mydict.keys()
  count = 0
  if valfold == -1: 
    valfold = (fold+1) % totalfold
  print('valfold'+str(valfold))
  trainls, valls, testls = [], [], []
  for train, test in skf.split(x,y):
    print(len(train), len(test))
    if count == fold:
      #print test[:]
      testls = test[:]
    elif count == valfold:
      valls = test[:]
    else:
      for i in test:
        trainls.append(i)
    count += 1
  return trainls, valls, testls

def loadim(fname, preprocesspath=preprocesspath):
  ''' from preprocess path load fname
  fname file name in preprocesspath
  aug is true, we augment im fliplr, rot 4'''
  ims = []
  with open(join(preprocesspath, fname), 'rb') as inputfile:
    im = cPickle.load(inputfile)
    #up_bound = np.random.choice(174)                          #zero out square
    #right_bound = np.random.choice(174)
    img = im
    #img[up_bound:(up_bound+50), right_bound:(right_bound+50)] = 0.0
    ims.append(img)
    inputfile.close()
  return ims

def loaddata(fold, totalfold, usedream=True, aug=True):
  '''get the fold th train and  test data from inbreast
  fold is the returned fold th data, from 0 to totalfold-1
  total fold is for the cross validation'''
  mydict = readlabel()
  mydictkey = mydict.keys()
  mydictvalue = mydict.values()
  trainindex, testindex = cvsplit(fold, totalfold, mydict)
  if aug == True:
    traindata, trainlabel = np.zeros((6*len(trainindex),227,227)), np.zeros((6*len(trainindex),))
  else:
    traindata, trainlabel = np.zeros((len(trainindex),227,227)), np.zeros((len(trainindex),))
  testdata, testlabel =  np.zeros((len(testindex),227,227)), np.zeros((len(testindex),))
  traincount = 0
  for i in xrange(len(trainindex)):
    ims = loadim(mydictkey[trainindex[i]]+'.pickle', aug=aug)
    for im in ims:
      traindata[traincount, :, :] = im
      trainlabel[traincount] = mydictvalue[trainindex[i]]
      traincount += 1
  assert(traincount==traindata.shape[0])
  testcount = 0
  for i in xrange(len(testindex)):
    ims = loadim(mydictkey[testindex[i]]+'.pickle', aug=aug)
    testdata[testcount,:,:] = ims[0]
    testlabel[testcount] = mydictvalue[testindex[i]]
    testcount += 1
  assert(testcount==testdata.shape[0])
  if usedream:
    outx, outy = extractdreamdata()
    traindata = np.concatenate((traindata,outx), axis=0)
    trainlabel = np.concatenate((trainlabel,outy), axis=0)
  return traindata, trainlabel, testdata, testlabel

def loaddataenhance(fold, totalfold, valfold=-1, valnum=60):
  '''get the fold th train and  test data from inbreast
  fold is the returned fold th data, from 0 to totalfold-1
  total fold is for the cross validation'''
  mydict = readlabel()
  mydictkey = mydict.keys()
  mydictvalue = mydict.values()
  trainindex, valindex, testindex = cvsplitenhance(fold, totalfold, mydict, valfold=valfold)
  traindata, trainlabel = np.zeros((len(trainindex),227,227)), np.zeros((len(trainindex),))
  valdata, vallabel =  np.zeros((len(valindex),227,227)), np.zeros((len(valindex),))
  testdata, testlabel =  np.zeros((len(testindex),227,227)), np.zeros((len(testindex),))
  traincount = 0
  for i in xrange(len(trainindex)):
    ims = loadim(mydictkey[trainindex[i]]+'227.pickle')
    for im in ims:
      traindata[traincount, :, :] = im
      trainlabel[traincount] = int(mydictvalue[trainindex[i]])
      traincount += 1
  assert(traincount==traindata.shape[0])
  valcount = 0
  for i in xrange(len(valindex)):
    ims = loadim(mydictkey[valindex[i]]+'227.pickle')
    valdata[valcount,:,:] = ims[0]
    vallabel[valcount] = int(mydictvalue[valindex[i]])
    valcount += 1
  assert(valcount==valdata.shape[0])
  testcount = 0
  for i in xrange(len(testindex)):
    #print mydictkey[testindex[i]]
    ims = loadim(mydictkey[testindex[i]]+'227.pickle')
    testdata[testcount,:,:] = ims[0]
    testlabel[testcount] = int(mydictvalue[testindex[i]])
    testcount += 1
  assert(testcount==testdata.shape[0])
  #print(valdata.shape)
  randindex = np.random.permutation(valdata.shape[0])
  valdata = valdata[randindex,:,:]
  vallabel = vallabel[randindex]
  #print(valdata.shape)
  traindata = np.concatenate((traindata, valdata[valnum:,:,:]), axis=0)
  trainlabel = np.concatenate((trainlabel, vallabel[valnum:]), axis=0)
  valdata = valdata[:valnum,:,:]
  vallabel = vallabel[:valnum]
  maxvalue = (traindata.max()*1.0)
  print('inbreast max %f', maxvalue)
  traindata = traindata / maxvalue
  valdata = valdata / maxvalue
  testdata = testdata / maxvalue
  print('train data feature')
  #meanx = traindata.mean()
  #stdx = traindata.std()
  #traindata -= meanx
  #traindata /= stdx
  #valdata -= meanx
  #valdata /= stdx
  #testdata -= meanx
  #testdata /= stdx
  print(traindata.mean(), traindata.std(), traindata.max(), traindata.min())
  print('val data feature')
  print(valdata.mean(), valdata.std(), valdata.max(), valdata.min())
  print('test data feature')
  print(testdata.mean(), testdata.std(), testdata.max(), testdata.min())
  #meandata = traindata.mean()
  #stddata = traindata.std()
  #traindata = traindata - meandata
  #traindata = traindata / stddata
  #valdata = valdata - meandata
  #valdata = valdata / stddata
  #testdata = testdata - meandata
  #testdata = testdata / stddata
  return traindata, trainlabel, valdata, vallabel, testdata, testlabel

if __name__ == '__main__':
  traindata, trainlabel, testdata, testlabel = loaddata(0, 5)
  print(sum(trainlabel), sum(testlabel))

  traindata, trainlabel, valdata, vallabel, testdata, testlabel = loaddataenhance(0, 5)
  print(sum(trainlabel), sum(vallabel), sum(testlabel))
