import moderngl
import numpy as np

from manimlib.mobject.mobject import Mobject
from manimlib.utils.bezier import inverse_interpolate
from manimlib.utils.color import get_color_map
from manimlib.utils.iterables import resize_with_interpolation


class HeatMap(Mobject):
    """
    A mobject that displays a 2D numpy array as a heatmap.
    """

    # Use triangles for rendering since we'll create a grid of triangles
    render_primitive: int = moderngl.TRIANGLES
    shader_folder: str = "surface"  # Use surface shaders for proper coloring

    def __init__(
        self,
        data: np.ndarray,  # 2D numpy array
        color_map: str = "3b1b_colormap",  # Default colormap
        vmin: float | None = None,
        vmax: float | None = None,
        width: float = 4.0,
        height: float = 4.0,
        **kwargs,
    ):
        self.data = np.asarray(data)
        self.color_map_name = color_map
        self.color_map = get_color_map(color_map)

        # Set value range for color mapping
        self.vmin = vmin if vmin is not None else self.data.min()
        self.vmax = vmax if vmax is not None else self.data.max()

        # Store dimensions
        self.width = width
        self.height = height

        # Call parent constructor
        super().__init__(**kwargs)

    def init_points(self):
        """
        Create a grid of points based on the data dimensions.
        Each data point becomes a vertex in the grid.
        """
        rows, cols = self.data.shape

        # Create grid of points in 3D space
        # x ranges from -width/2 to width/2
        # y ranges from -height/2 to height/2
        # z is 0 for all points (2D heatmap)
        x_range = np.linspace(-self.width / 2, self.width / 2, cols)
        y_range = np.linspace(-self.height / 2, self.height / 2, rows)

        # Create meshgrid
        X, Y = np.meshgrid(x_range, y_range)

        # Flatten and combine into points array
        points = np.zeros((rows * cols, 3))
        points[:, 0] = X.flatten()  # x coordinates
        points[:, 1] = Y.flatten()  # y coordinates
        points[:, 2] = 0  # z coordinates (all 0 for 2D)

        # Set the points
        self.set_points(points)

        # Create triangle indices for rendering
        self.compute_triangle_indices(rows, cols)

        # Set colors based on data values
        self.update_colors()

    def compute_triangle_indices(self, rows: int, cols: int):
        """
        Create triangle indices for rendering a grid.
        Each cell in the grid is made of two triangles.
        """
        if rows == 0 or cols == 0:
            self.triangle_indices = np.zeros(0, dtype=int)
            return

        # Create index grid
        index_grid = np.arange(rows * cols).reshape((rows, cols))

        # Create indices for triangles (two triangles per grid cell)
        indices = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                # First triangle: top-left, bottom-left, top-right
                indices.extend(
                    [
                        index_grid[i, j],  # top-left
                        index_grid[i + 1, j],  # bottom-left
                        index_grid[i, j + 1],  # top-right
                    ]
                )
                # Second triangle: top-right, bottom-left, bottom-right
                indices.extend(
                    [
                        index_grid[i, j + 1],  # top-right
                        index_grid[i + 1, j],  # bottom-left
                        index_grid[i + 1, j + 1],  # bottom-right
                    ]
                )

        self.triangle_indices = np.array(indices, dtype=int)

    def get_triangle_indices(self) -> np.ndarray:
        """Return triangle indices for rendering."""
        return self.triangle_indices

    def update_colors(self):
        """
        Update colors based on data values using the color map.
        """
        if not self.has_points():
            return

        # Normalize data values to [0, 1] range
        normalized = inverse_interpolate(self.vmin, self.vmax, self.data.flatten())
        normalized = np.clip(normalized, 0, 1)

        # Get colors from color map
        colors = self.color_map(normalized)

        # Set rgba values for all points
        self.data["rgba"][:] = colors

    def set_data(
        self, new_data: np.ndarray, vmin: float | None = None, vmax: float | None = None
    ):
        """
        Update the heatmap with new data.
        """
        self.data = np.asarray(new_data)

        # Update value range if provided
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax

        # Update colors
        self.update_colors()
        self.note_changed_data()
        return self

    def set_color_map(self, color_map: str):
        """
        Change the color map used for the heatmap.
        """
        self.color_map_name = color_map
        self.color_map = get_color_map(color_map)
        self.update_colors()
        self.note_changed_data()
        return self

    def set_value_range(self, vmin: float, vmax: float):
        """
        Set the value range for color mapping.
        """
        self.vmin = vmin
        self.vmax = vmax
        self.update_colors()
        self.note_changed_data()
        return self

    def get_value_at_position(self, x: float, y: float) -> float | None:
        """
        Get the data value at a specific position in the heatmap.
        Returns None if position is outside the heatmap bounds.
        """
        # Convert position to grid coordinates
        grid_x = (x + self.width / 2) / self.width * (self.data.shape[1] - 1)
        grid_y = (y + self.height / 2) / self.height * (self.data.shape[0] - 1)

        # Check bounds
        if (
            grid_x < 0
            or grid_x >= self.data.shape[1] - 1
            or grid_y < 0
            or grid_y >= self.data.shape[0] - 1
        ):
            return None

        # Bilinear interpolation
        x0 = int(grid_x)
        y0 = int(grid_y)
        x1 = min(x0 + 1, self.data.shape[1] - 1)
        y1 = min(y0 + 1, self.data.shape[0] - 1)

        dx = grid_x - x0
        dy = grid_y - y0

        # Interpolate
        top = self.data[y0, x0] * (1 - dx) + self.data[y0, x1] * dx
        bottom = self.data[y1, x0] * (1 - dx) + self.data[y1, x1] * dx
        return top * (1 - dy) + bottom * dy

    def add_color_bar(self, width: float = 0.2, height: float = 4.0, buff: float = 0.5):
        """
        Create a color bar for the heatmap.
        Returns a separate mobject that can be added to the scene.
        """
        from manimlib.mobject.geometry import Rectangle
        from manimlib.mobject.types.vectorized_mobject import VGroup

        # Create gradient rectangle
        color_bar = Rectangle(width=width, height=height, stroke_width=0)

        # Position it to the right of the heatmap
        color_bar.next_to(self, RIGHT, buff=buff)

        # Create gradient colors
        n_samples = 100
        values = np.linspace(0, 1, n_samples)
        colors = self.color_map(values)

        # Set gradient (this would need custom implementation)
        # For simplicity, we'll create a gradient using multiple rectangles
        gradient = VGroup()
        for i in range(n_samples):
            segment = Rectangle(
                width=width,
                height=height / n_samples,
                fill_color=rgb_to_hex(colors[i, :3]),
                fill_opacity=colors[i, 3],
                stroke_width=0,
            )
            segment.shift(UP * (height / 2 - (i + 0.5) * height / n_samples))
            gradient.add(segment)

        # Replace the solid rectangle with gradient
        color_bar.become(gradient)

        return color_bar
