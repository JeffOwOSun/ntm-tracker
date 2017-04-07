# network definitions, in [kernelsize, relative stride, padding, dilation]
VGGDef = [
        [3, 1, 1, 1, 'conv1_1'],
        [3, 1, 1, 1, 'conv1_2'],
        [2, 2, 0, 1, 'pool1'],
        [3, 1, 1, 1, 'conv2_1'],
        [3, 1, 1, 1, 'conv2_2'],
        [2, 2, 0, 1, 'pool2'],
        [3, 1, 1, 1, 'conv3_1'],
        [3, 1, 1, 1, 'conv3_2'],
        [3, 1, 1, 1, 'conv3_3'],
        [2, 2, 0, 1, 'pool3'],
        [3, 1, 1, 1, 'conv4_1'],
        [3, 1, 1, 1, 'conv4_2'],
        [3, 1, 1, 1, 'conv4_3'],
        ]
'''
calculate the receptivefield size, absolute stride size, blob size
'''
def calcRF(netDef=VGGDef, inputSize=300):
    r = 1 #receptive field
    s = 1 #absolute stride
    output = [[r,s,inputSize,'data']]
    for layer in netDef:
        k, rs, p, d, name = layer #kernel size and relative stride
        r = r + d * (k - 1) * s # receptive field r_i = r_i-1 + (k_i - 1) * s_i-1
        s = s * rs # absolute stride s_i = s_i-1 * rs_i
        inputSize = (inputSize - (d * (k - 1) + 1) + 2*p)/rs + 1
        output.append([r, s, inputSize, name])
    return output

'''
project the receptive field of the given pixel (layer, x, y) down to input space

layerNo is in the scale of number of feature maps, which is 1 larger than number
of kernels.
'''
def projectRF(layerNo=0, x=0, y=0, netDef=VGGDef, inputSize=300, layers=None):
    '''first calculate the layer info. The only thing needed is blob size'''
    if not layers:
        layers = calcRF(netDef, inputSize)
    boxes = []
    '''
    1. find the four corner pixel on the previous layer
    2. for each corner pixel, only find the one pixel on the same direction
    there after
    '''
    while layerNo < 0:
        layerNo += len(layers)
    #current layer
    corners = [(x,y)]*4 #the four corner points are the same point as (x,y)
    boxes.append({
        'name': layers[layerNo][-1],
        'receptiveField': layers[layerNo][0],
        'absoluteStride': layers[layerNo][1],
        'blobSize': layers[layerNo][2],
        'corners': corners
        })

    '''project the first layer, generate the first four corner points'''
    kernelSize, relativeStride, padding, dilation, name = netDef[layerNo-1] #This layer's kernel size info
    kernelSize = (kernelSize - 1) * dilation + 1
    prevBlobSize = layers[layerNo-1][2] #previous layer's blobSize, used to constrain coordinates
    if x < 0 or x >= prevBlobSize or y < 0 or y >= prevBlobSize:
        raise Exception('x,y range illegal: {}'.format((x,y)))
    corners = rectify([
        (0-padding+x*relativeStride, 0-padding+y*relativeStride), #left top corner
        (0-padding+kernelSize-1+x*relativeStride, 0-padding+y*relativeStride), #right top corner
        (0-padding+kernelSize-1+x*relativeStride, 0-padding+kernelSize-1+y*relativeStride), #right bottom corner
        (0-padding+x*relativeStride, 0-padding+kernelSize-1+y*relativeStride), #left bottom corner
        ], prevBlobSize)
    boxes.append({
        'name': layers[layerNo-1][-1], #name of feature map, as marked by outputing kernel name
        'receptiveField': layers[layerNo-1][0], #receptive field size of feature map
        'absoluteStride': layers[layerNo-1][1], #absolute stride size of feature map
        'blobSize': layers[layerNo-1][2], #blobSize of THIS feature map
        'corners': corners
        })
    layerNo -= 1

    '''now propagate the corners down to the input layer'''
    while layerNo > 0:
        kernelSize, relativeStride, padding, dilation, name = netDef[layerNo-1]
        kernelSize = (kernelSize - 1) * dilation + 1
        prevBlobSize = layers[layerNo-1][2]
        corners = rectify([
            (0-padding+corners[0][0]*relativeStride, 0-padding+corners[0][1]*relativeStride), #left top corner
            (0-padding+kernelSize-1+corners[1][0]*relativeStride, 0-padding+corners[1][1]*relativeStride), #right top corner
            (0-padding+kernelSize-1+corners[2][0]*relativeStride, 0-padding+kernelSize-1+corners[2][1]*relativeStride), #right bottom corner
            (0-padding+corners[3][0]*relativeStride, 0-padding+kernelSize-1+corners[3][1]*relativeStride), #left bottom corner
            ], prevBlobSize)
        boxes.append({
            'name': layers[layerNo-1][-1],
            'receptiveField': layers[layerNo-1][0],
            'absoluteStride': layers[layerNo-1][1],
            'blobSize': layers[layerNo-1][2],
            'corners': corners
            })
        layerNo -= 1

    boxes.reverse() #reverse the order so that the data layer is at the smallest index
    return boxes, layers

def projectLayers(layersToProject=['conv4_3', 'pool6'], netDef=VGGDef, inputSize=300):
    '''first calculate the layer info. The only thing needed is blob size'''
    layers = calcRF(netDef, inputSize)
    '''put the layer informatin into a dictionary'''
    layerDict = {}
    for idx, layer in enumerate(layers):
        layerDict[layer[-1]] = layer+[idx]

    boxesByLayers = {}
    for l in layersToProject:
        receptiveField, absoluteStride, blobSize, name, idx = layerDict[l]
        boxes = []
        '''row-major order'''
        for y in xrange(blobSize):
            for x in xrange(blobSize):
                ret, _ = projectRF(idx, x, y, netDef=VGGDef,
                        inputSize=inputSize, layers=layers)
                corners = ret[0]['corners']
                xmin = corners[0][0]
                ymin = corners[0][1]
                xmax = corners[2][0]
                ymax = corners[2][1]
                boxes.append({'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax})
        boxesByLayers[l] = boxes

    return boxesByLayers


def rectify(box, blobSize):
    return [tuple(min(blobSize-1, max(val, 0)) for val in point) for point in box]

conv43Points = [
        (6,6), (6,8), (6,10), (6,12), (6,14), (6,16), (6,18), (6,20),
        (8,6), (8,8), (8,10), (8,12), (8,14), (8,16), (8,18), (8,20),
        (10,6), (10,8), (10,10), (10,12), (10,14), (10,16), (10,18), (10,20),
        (12,6), (12,8), (12,10), (12,12), (12,14), (12,16), (12,18), (12,20),
        (14,6), (14,8), (14,10), (14,12), (14,14), (14,16), (14,18), (14,20),
        (16,6), (16,8), (16,10), (16,12), (16,14), (16,16), (16,18), (16,20),
        (18,6), (18,8), (18,10), (18,12), (18,14), (18,16), (18,18), (18,20),
        (20,6), (20,8), (20,10), (20,12), (20,14), (20,16), (20,18), (20,20),]

def get_receptive_fields(points=conv43Points, layer='conv4_3'):
    rfs = {}
    for x,y in points:
        boxes, layers = projectRF(-1, x, y, netDef=VGGDef, inputSize=224)
        for box in boxes:
            if box['name'] == layer:
                rfs[(x,y)] = box['corners']
    return rfs

if __name__ == '__main__':
    #for row in calcRF():
    #    print("layer {}, receptive field {}, absolute stride {}, output blob width {}".format(row[-1],row[0],row[1],row[2]))

    layerNo = -1
    x = 0
    y = 0
    while True:
        x = int(raw_input('x:'))
        y = int(raw_input('y:'))
        boxes, layers = projectRF(layerNo, x, y, netDef=VGGDef, inputSize=224)
        for idx, box in enumerate(boxes):
            print 'layer={} receptiveField={} absoluteStride={} blob={} corners={}'.format(box['name'], box['receptiveField'], box['absoluteStride'], box['blobSize'], box['corners'])
