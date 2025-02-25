import sys
from trame.app import get_server
from trame.widgets import vuetify, vtk as trame_vtk
from trame.ui.vuetify import SinglePageLayout
import numpy as np
import vtkmodules.all as vtk
from matplotlib import colormaps, colors
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """Set the image actor for interaction."""
        self.image_actor = actor
        if actor:
            actor.SetPosition(0, 0, 0)  # Set initial position
    
    def OnLeftButtonDown(self, obj, event):
        """Start panning when the left mouse button is pressed."""
        self.pan_mode = True
        self.last_x, self.last_y = self.GetInteractor().GetEventPosition()
    
    def OnLeftButtonUp(self, obj, event):
        """Stop panning when the left mouse button is released."""
        self.pan_mode = False
    
    def OnMouseMove(self, obj, event):
        """Handle panning by updating the actor's position."""
        if self.pan_mode and self.image_actor:
            x, y = self.GetInteractor().GetEventPosition()
            old_pos = self.image_actor.GetPosition()
            dx = x - self.last_x
            dy = y - self.last_y
            self.image_actor.SetPosition(old_pos[0] + dx, old_pos[1] - dy, old_pos[2])
            self.last_x, self.last_y = x, y
            self.GetInteractor().Render()
    
    def OnMouseWheelForward(self, obj, event):
        """Zoom in when the mouse wheel is scrolled forward."""
        if self.image_actor:
            self.zoom_level = min(5.0, self.zoom_level * 1.2)
            self._update_zoom()
    
    def OnMouseWheelBackward(self, obj, event):
        """Zoom out when the mouse wheel is scrolled backward."""
        if self.image_actor:
            self.zoom_level = max(0.1, self.zoom_level / 1.2)
            self._update_zoom()
    
    def _update_zoom(self):
        """Update the actor's scale to reflect the zoom level."""
        if not self.image_actor:
            return
        self.image_actor.SetScale(self.zoom_level, self.zoom_level, 1)
        self.GetInteractor().Render()
    
    def ResetView(self):
        """Reset view to initial state"""
        if self.image_actor:
            self.zoom_level = 1.0
            self.image_actor.SetPosition(0, 0, 0)
            self._update_zoom()

def load_matrix_from_file(filename):
    """Load matrix from file with error checking"""
    logger.info(f"Loading matrix from {filename}")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Matrix file {filename} not found.")
    
    matrix = np.loadtxt(filename, delimiter=' ')
    logger.info(f"Loaded matrix shape: {matrix.shape}")
    return matrix

def create_vtk_pipeline(matrix, colormap_name="YlOrRd"):
    """Create VTK visualization pipeline"""
    size = matrix.shape[0]
    logger.info(f"Creating visualization for {size}x{size} matrix")
    
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(size, size, 1)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    
    colormap = colormaps[colormap_name]
    norm = colors.Normalize(vmin=0, vmax=1)
    
    matrix_float = matrix.astype(float)
    rgb_values = (colormap(norm(matrix_float)) * 255).astype(np.uint8)
    
    # Populate image data
    for i in range(size):
        for j in range(size):
            if matrix[i, j] > 0:
                r, g, b = rgb_values[i, j, :3]
            else:
                r, g, b = 245, 245, 245
            
            image_data.SetScalarComponentFromDouble(j, i, 0, 0, int(r))
            image_data.SetScalarComponentFromDouble(j, i, 0, 1, int(g))
            image_data.SetScalarComponentFromDouble(j, i, 0, 2, int(b))
    
    actor = vtk.vtkImageActor()
    actor.SetInputData(image_data)
    
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)
    
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    
    return actor, renderer, render_window

def calculate_matrix_stats(matrix):
    """Calculate matrix statistics"""
    size = matrix.shape[0]
    total_edges = np.sum(matrix) // 2
    max_possible_edges = (size * (size - 1)) // 2
    density = total_edges / max_possible_edges
    num_communities = size // 20
    
    return f"""
    Matrix Size: {size}x{size}
    Total Edges: {total_edges:,}
    Density: {density:.2%}
    Number of Communities: {num_communities}
    """

def setup_server(matrix_file):
    """Setup and configure the Trame server"""
    server = get_server(client_type="vue2")
    
    # Initialize state
    state = server.state
    state.trame__title = "Matrix Visualization"
    state.current_matrix_file = matrix_file
    state.colormap_name = "YlOrRd"
    state.matrix_stats = ""
    state.loading = False
    state.error_message = ""
    
    try:
        # Load matrix from provided file
        matrix = load_matrix_from_file(matrix_file)
        
        # Create visualization pipeline
        actor, renderer, render_window = create_vtk_pipeline(
            matrix,
            state.colormap_name
        )
        
        # Setup VTK interactor
        render_window.Render()
        interactor = vtk.vtkRenderWindowInteractor()
        render_window.SetInteractor(interactor)
        
        style = CustomInteractorStyle()
        interactor.SetInteractorStyle(style)
        
        interactor.Initialize()
        style.SetImageActor(actor)
        
        render_window.Render()
        
        # Create Trame layout
        with SinglePageLayout(server) as layout:
            layout.title.set_text(f"Matrix Visualization - {os.path.basename(matrix_file)}")
            
            with layout.toolbar:
                vuetify.VSelect(
                    label="Color Scheme",
                    items=[
                        {"text": "Yellow-Orange-Red", "value": "YlOrRd"},
                        {"text": "Red-Purple", "value": "RdPu"},
                        {"text": "Purple-Blue", "value": "PuBu"},
                        {"text": "Orange-Red", "value": "OrRd"},
                        {"text": "Blue-Purple", "value": "BuPu"},
                        {"text": "Heat", "value": "hot"},
                        {"text": "Viridis", "value": "viridis"},
                        {"text": "Plasma", "value": "plasma"}
                    ],
                    v_model=("colormap_name",),
                    dense=True,
                )
                
                vuetify.VSpacer()
                vuetify.VBtn("Zoom In", click="trame.trigger('zoom_in')")
                vuetify.VBtn("Zoom Out", click="trame.trigger('zoom_out')")
                vuetify.VBtn("Reset View", click="trame.trigger('reset_view')")
                
            # Main content
            with layout.content:
                with vuetify.VContainer(fluid=True):
                    # Error message display
                    with vuetify.VRow(v_show="error_message"):
                        with vuetify.VCol(cols=12):
                            vuetify.VAlert(
                                text=("error_message",),
                                type="error",
                                dismissible=True,
                            )
                    
                    # Main content row
                    with vuetify.VRow():
                        # Matrix visualization
                        with vuetify.VCol(cols=9):
                            trame_vtk.VtkRemoteView(render_window, ref="vtk_view")
                            vuetify.VProgressLinear(
                                indeterminate=True,
                                active=("loading",),
                                absolute=True,
                                bottom=True
                            )
                        
                        # Side panel
                        with vuetify.VCol(cols=3):
                            with vuetify.VCard(classes="pa-3 mb-3"):
                                vuetify.VCardTitle("Matrix Statistics")
                                vuetify.VCardText(v_text=("matrix_stats",))
                            
                            with vuetify.VCard(classes="pa-3"):
                                vuetify.VCardTitle("Color Legend")
                                vuetify.VCardText("""
                                - Dark Color: Strong connection
                                - Light Color: Weak connection
                                - Light Gray: No connection
                                
                                Use mouse wheel to zoom
                                Click and drag to pan
                                """)
        
        @server.state.change("colormap_name")
        def update_colormap_name(colormap_name, **kwargs):
            nonlocal actor, renderer, render_window
            state.loading = True
            try:
                actor, renderer, render_window = create_vtk_pipeline(
                    matrix,
                    colormap_name=colormap_name
                )
            except Exception as e:
                logger.error(f"Error updating colormap: {e}")
            finally:
                state.loading = False
            server.trigger("vtk_view_update")
        
        @server.state.change("zoom_in")
        def zoom_in():
            style.OnMouseWheelForward(None, None)
        
        @server.state.change("zoom_out")
        def zoom_out():
            style.OnMouseWheelBackward(None, None)
        
        @server.state.change("reset_view")
        def reset_view():
            style.ResetView()
        
        # Initialize statistics
        state.matrix_stats = calculate_matrix_stats(matrix)
        
    except Exception as e:
        error_msg = f"Error during initialization: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
    
    return server

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 matrix_viz.py <matrix_file>")
        print("Examples:")
        print("  python3 matrix_viz.py matrix_100.txt")
        print("  python3 matrix_viz.py matrix_500.txt")
        print("  python3 matrix_viz.py matrix_1000.txt")
        sys.exit(1)
    
    matrix_file = sys.argv[1]
    
    if not os.path.exists(matrix_file):
        print(f"Error: Matrix file '{matrix_file}' not found.")
        print("Please run matrix_gen.py first to generate the matrix files.")
        sys.exit(1)
    
    server = setup_server(matrix_file)
    server.start()