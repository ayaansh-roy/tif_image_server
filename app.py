from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from PIL import Image
import numpy as np
import rasterio
import mercantile
import io

app = FastAPI()

raster_path = r'D:\dev\aiml\pooja_projects\raster_image_server\CarissaMacrocarpa_files\CarissaMacrocarpa_2100_ssp370_Ensemble.tif'
raster = rasterio.open(raster_path)

# Prepare coordinate transformer once
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def get_tile_1(z, x, y, tile_size=256):
    tile_bounds_3857 = mercantile.xy_bounds(x, y, z)
    left, bottom = transformer.transform(tile_bounds_3857.left, tile_bounds_3857.bottom)
    right, top = transformer.transform(tile_bounds_3857.right, tile_bounds_3857.top)

    print(f"Tile bounds EPSG:4326 at z={z}, x={x}, y={y}: {left}, {bottom}, {right}, {top}")

    try:
        window = from_bounds(left, bottom, right, top, transform=raster.transform)
        window = window.round_offsets(op='floor').round_lengths(op='ceil')

        if window.width == 0 or window.height == 0:
            raise ValueError("Tile is outside raster bounds")

        data = raster.read(1, window=window, out_shape=(tile_size, tile_size), resampling=Resampling.bilinear)

        data = np.clip(data, 0, np.percentile(data, 98))
        data_min, data_max = data.min(), data.max()

        if data_max - data_min == 0:
            norm_data = np.zeros_like(data, dtype=np.uint8)
        else:
            norm_data = (255 * (data - data_min) / (data_max - data_min)).astype(np.uint8)

        img = Image.fromarray(norm_data).convert("L")
        return img.convert("RGBA")

    except Exception as e:
        print(f"Tile fetch error at z={z}, x={x}, y={y}: {e}")
        # Return transparent tile
        return Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))


from PIL import Image

def get_tile_2(z, x, y, tile_size=256):
    import mercantile
    from pyproj import Transformer
    import numpy as np
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling

    tile_bounds_3857 = mercantile.xy_bounds(x, y, z)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    left, bottom = transformer.transform(tile_bounds_3857.left, tile_bounds_3857.bottom)
    right, top = transformer.transform(tile_bounds_3857.right, tile_bounds_3857.top)

    try:
        window = from_bounds(left, bottom, right, top, transform=raster.transform)
        window = window.round_offsets(op='floor').round_lengths(op='ceil')

        if window.width == 0 or window.height == 0:
            raise ValueError("Tile outside raster bounds")

        data = raster.read(1, window=window, out_shape=(tile_size, tile_size), resampling=Resampling.bilinear)

        # Check if data is all nodata
        if np.all((data == raster.nodata) | np.isnan(data)):
            raise ValueError("Tile contains only NoData")

        data = np.clip(data, 0, np.percentile(data, 98))
        data_min = data.min()
        data_max = data.max()

        if data_max - data_min == 0:
            data[:] = 0
        else:
            data = (255 * (data - data_min) / (data_max - data_min)).astype(np.uint8)

        return Image.fromarray(data).convert("L")

    except Exception as e:
        print(f"Tile fetch error at z={z}, x={x}, y={y}: {e}")
        # Return fully transparent tile (RGBA)
        return Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))


@app.get("/tiles/{z}/{x}/{y}.png")
def tile(z: int, x: int, y: int):
    img = get_tile_2(z, x, y)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
