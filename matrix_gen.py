from trame.app import get_server
from trame.widgets import vuetify, vtk as trame_vtk
from trame.ui.vuetify import SinglePageLayout
import numpy as np
import vtkmodules.all as vtk

# Custom interactor style to support panning and zooming
class CustomInteractorStyle(vtk.vtkInteractorStyleImage):
    def __init__(self):
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self.OnLeftButtonDown)
        self.AddObserver("LeftButtonReleaseEvent", self.OnLeftButtonUp)
        self.AddObserver("MouseMoveEvent", self.OnMouseMove)
        self.AddObserver("MouseWheelForwardEvent", self.OnMouseWheelForward)
        self.AddObserver("MouseWheelBackwardEvent", self.OnMouseWheelBackward)
        
        self.pan_mode = False
        self.last_x = 0
        self.last_y = 0
        self.image_actor = None
        self.zoom_level = 1.0

    def SetImageActor(self, actor):
        self.image_actor = actor
        actor.SetPosition(0, 0)

    def OnLeftButtonDown(self, obj, event):
        self.pan_mode = True
        self.last_x, self.last_y = self.GetInteractor().GetEventPosition()

    def OnLeftButtonUp(self, obj, event):
        self.pan_mode = False

    def OnMouseMove(self, obj, event):
        if self.pan_mode and self.image_actor:
            x, y = self.GetInteractor().GetEventPosition()
            old_pos = self.image_actor.GetPosition()
            dx = x - self.last_x
            dy = y - self.last_y
            self.image_actor.SetPosition(old_pos[0] + dx, old_pos[1] - dy)  # Adjust for VTK's coordinate system
            self.last_x, self.last_y = x, y
            self.GetInteractor().Render()

    def OnMouseWheelForward(self, obj, event):
        if self.image_actor:
            self.zoom_level *= 1.2
            self._update_zoom()

    def OnMouseWheelBackward(self, obj, event):
        if self.image_actor:
            self.zoom_level /= 1.2
            self._update_zoom()

    def _update_zoom(self):
        if not self.image_actor:
            return
        self.image_actor.SetScale(self.zoom_level, self.zoom_level, 1)
        self.GetInteractor().Render()


# Function to generate a sample matrix
def generate_matrix(size=100, num_communities=5, community_density=0.7, inter_community_density=0.1):
    matrix = np.zeros((size, size), dtype=int)
    community_size = size // num_communities
    for i in range(num_communities):
        start_idx = i * community_size
        end_idx = start_idx + community_size
        block = np.random.random((community_size, community_size)) < community_density
        block = np.triu(block, 1) + np.triu(block, 1).T
        matrix[start_idx:end_idx, start_idx:end_idx] = block.astype(int)
    for i in range(size):
        for j in range(i + 1, size):
            if (i // community_size) != (j // community_size) and np.random.random() < inter_community_density:
                matrix[i, j] = matrix[j, i] = 1
    return matrix


# Create the VTK pipeline
def create_vtk_pipeline(matrix):
    size = matrix.shape[0]
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(size, size, 1)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    for i in range(size):
        for j in range(size):
            if matrix[i, j] > 0:
                image_data.SetScalarComponentFromDouble(j, i, 0, 0, 0)
                image_data.SetScalarComponentFromDouble(j, i, 0, 1, 0)
                image_data.SetScalarComponentFromDouble(j, i, 0, 2, 255)
            else:
                image_data.SetScalarComponentFromDouble(j, i, 0, 0, 255)
                image_data.SetScalarComponentFromDouble(j, i, 0, 1, 255)
                image_data.SetScalarComponentFromDouble(j, i, 0, 2, 255)
    actor = vtk.vtkImageActor()
    actor.SetInputData(image_data)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    return actor, renderer, render_window


# Create Trame server and VTK visualization
def setup_server():
    server = get_server(client_type="vue2")
    matrix = generate_matrix(size=100)
    actor, renderer, render_window = create_vtk_pipeline(matrix)
    interactor = render_window.GetInteractor()
    style = CustomInteractorStyle()
    style.SetImageActor(actor)
    interactor.SetInteractorStyle(style)
    with SinglePageLayout(server) as layout:
        layout.title.set_text("Interactive Graph Visualization")
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VBtn("Zoom In", click="trame.trigger('zoom_in')")
            vuetify.VBtn("Zoom Out", click="trame.trigger('zoom_out')")
        with layout.content:
            trame_vtk.VtkRemoteView(render_window, ref="vtk_view")

    @server.state.change("zoom_in")
    def zoom_in():
        style.zoom_level *= 1.2
        style._update_zoom()

    @server.state.change("zoom_out")
    def zoom_out():
        style.zoom_level /= 1.2
        style._update_zoom()

    return server


if __name__ == "__main__":
    server = setup_server()
    server.start()
