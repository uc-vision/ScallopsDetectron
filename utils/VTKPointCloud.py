import vtk
from vtk.util import numpy_support
import numpy as np


class VtkPointCloud:
    def __init__(self, max_num_points=1e6, pnt_size=2):
        self.maxNumPoints = max_num_points
        self.vtkPolyData = vtk.vtkPolyData()
        self.vtkPoints = vtk.vtkPoints()
        self.vtkColors = vtk.vtkUnsignedCharArray()
        self.vtkColors.SetName("Colors")
        self.vtkColors.SetNumberOfComponents(3)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.GetProperty().SetPointSize(pnt_size)
        self.vtkActor.SetMapper(mapper)

    def setPoints(self, points_xyz, points_rgb):
        points_polyd = vtk.vtkPolyData()
        vtk_points = numpy_support.numpy_to_vtk(points_xyz)
        self.vtkPoints.SetData(vtk_points)
        vtk_pntcols = numpy_support.numpy_to_vtk((points_rgb * 255).astype(np.uint8))
        self.vtkColors = vtk_pntcols
        points_polyd.SetPoints(self.vtkPoints)
        vertexfilter = vtk.vtkVertexGlyphFilter()
        vertexfilter.SetInputData(points_polyd)
        vertexfilter.Update()
        self.vtkPolyData.ShallowCopy(vertexfilter.GetOutput())
        self.vtkPolyData.GetPointData().SetScalars(self.vtkColors)
        self.vtkPolyData.Modified()
