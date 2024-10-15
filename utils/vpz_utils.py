from shapely import *
import zipfile
import tempfile
import geopandas as gp
import xml.etree.cElementTree as ET

def get_shape_layers_gpd(dir_path, vpz_fn):
    zf = zipfile.ZipFile(dir_path + vpz_fn)
    shape_layers = []
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        vpz_root = ET.parse(tempdir + '/doc.xml').getroot()
        for child in vpz_root.iter('layer'):
            if child.attrib['type'] == 'shapes':
                elem_data = list(child.iter('data'))
                elem_src = list(child.iter('source'))
                if len(elem_data):
                    shape_fn = tempdir + '/' + elem_data[0].attrib['path']
                elif len(elem_src):
                    shape_fn = dir_path + elem_src[0].attrib['path']
                else:
                    continue
                try:
                    shape_layers.append([child.attrib['label'], gp.read_file(shape_fn)])
                except Exception as e:
                    print(e)
    return shape_layers