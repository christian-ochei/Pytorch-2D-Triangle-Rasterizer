import torch
import torch.nn as nn
import cv2

class TriRasterizer(nn.Module):
    def __init__(self,shape=(480,640),channels:int=3,_trainable=False):
        super(TriRasterizer, self).__init__()
        assert len(shape) == 2
        self._shape = shape
        self._backdrop = nn.Parameter(torch.zeros(*shape,channels,dtype=torch.float),requires_grad=_trainable)
        # Feel free arround with the quality
        self.quality = 0.1
        ...

    @staticmethod
    @torch.jit.script
    def _fill(poly,area,_type:torch.dtype=torch.float32):
        """
        The main part
        The algorithm takes the tensor points, repeats interleave them by their area
        Then for each tiangles in the batch:
            there should be the-area more triangles exactly the same
            the goal is for each of those extra triangles,
                interpolate each of those points such that: each points fills up the traingle

        """
        # calculate center of the base
        base_p = (poly[:,0] + poly[:,1])/2
        a_diff = (poly[:,2]-base_p) . detach()

        # then height
        heights = torch.sqrt_(torch.square_(a_diff[...,0]) + torch.square_(a_diff[...,1]))

        # comulative sum of the areas
        # This gives positional embeding for each points when subtracted from torch.arange
        cum   = torch.cumsum(area,dim=0).int()
        poly  = poly.type(_type)
        heights  = heights.type(torch.int16)
        poly    = poly.repeat_interleave(area,dim=0)
        cum_    = cum.repeat_interleave(area,dim=0)
        heights = heights.repeat_interleave(area,dim=0)

        # normalized makes the first point = 0 and last = 1 for the repeated points
        normalized = 1-(cum_-torch.arange(len(cum_)))/cum_

        # this is the exponential, it reduces pixels as it gets closer to the tip of the triangle
        exp = 1-torch.sqrt_(1-normalized)
        exp = exp.type(torch.float32)

        # This iterpolates more of the x as it gets to the tip of the triangle
        ij__interp = (exp % (torch.tensor(1,dtype=torch.float16)/heights))[...,None]
        # * heights[...,None] just brings the exponentials max back to 1
        # the random normal washes the points and makes it appear blurry
        ij_interp = ij__interp*heights[...,None] #+ torch.randn_like(ij__interp)/70

        # this makes the makes the c interpolation exponentialy faster
        k_interp = (exp[...,None] - ij__interp)  #+ torch.randn_like(ij__interp)/70

        # Finally interpolate them
        # weigh a and b first with respect to the calculated exponential
        ij_trunc = poly[:,0] * (1-ij_interp) + poly[:,1]*ij_interp
        # then weigh the output with c with respect to the modulus of the exponential
        filled   = poly[:,2] * k_interp  + ij_trunc * (1-k_interp)
        return filled

    def _compute_area(self,poly):
        # Formula to find area from a set of points
        area = torch.abs(0.5 * ((poly[..., 0, 0] * (poly[..., 1, 1] - poly[..., 2, 1])) +
                   (poly[...,1,0]*(poly[...,2,1]-poly[...,0,1])) +
                   (poly[...,2,0]*(poly[...,0,1]-poly[...,1,1]))))
        return area

    def forward(self,traingles,features):
        return self.rasterize(traingles,features)


    @torch.no_grad()
    def rasterize(self,traingles,features):
        raster = self._backdrop.clone()
        # first find the area
        # the lower the quality the more spaces in between pixels
        area = (self._compute_area(traingles)*self.quality).int()

        # fill in points based on computed area
        filled_cords = self._fill(traingles,area,_type=torch.int16)
        filled_cords = filled_cords.long()

        # clamp the points to prevent out of bounds exception
        filled_cords[...,0] = filled_cords[...,0].clamp(min=0,max=self._shape[1]-1)
        filled_cords[...,1] = filled_cords[...,1].clamp(min=0,max=self._shape[0]-1)

        # also fill in features based on computed area
        filled_feats = self._fill(features,area,_type=torch.float32)
        pixels = (filled_cords[...,1],filled_cords[...,0])

        # compute the times the same point has been placed on the raster
        sum_ = torch.zeros_like(raster).index_put(pixels,torch.ones_like(filled_feats),accumulate=True)

        # rasterize them using indexing
        raster = raster.index_put(pixels,filled_feats,accumulate=False)

        # average those sums
        raster = raster/sum_
        return raster
#

if __name__ == '__main__':
    tiangles = torch.tensor([
        [[100, 0], [0, 200], [20, 300]]
    ], dtype=torch.float) + 100

    print(tiangles.shape)
    tiangles = tiangles.expand(3,-1,-1).clone()

    features = torch.randn(*tiangles.shape[:-1], 3)
    rasterizer = TriRasterizer()
    j = torch.zeros(*tiangles.shape)

    while True:
        features = features*0.01 + torch.randn(*tiangles.shape[:-1], 3)*0.09
        j += torch.randn(*tiangles.shape)*0.1
        tiangles += j
        raster = rasterizer(tiangles,features+1)
        cv2.imshow('x',raster.numpy())
        cv2.waitKey(20)
