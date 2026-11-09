"""Publication-label checks preserve every source pixel and keep text inside its panel."""
from io import BytesIO
import unittest
from PIL import Image, ImageDraw
from research_pipeline.comparison_visuals import comparison_canvas


class ComparisonVisualTests(unittest.TestCase):
    def test_large_labels_and_complete_pixel_integrity_after_png_roundtrip(self):
        left=Image.new("RGB",(1920,1080),(23,91,143))
        right=Image.new("RGB",(1920,1080),(45,30,90))
        ImageDraw.Draw(left).rectangle((0,0,101,203),fill=(255,12,2))
        ImageDraw.Draw(right).rectangle((1800,900,1919,1079),fill=(1,233,87))
        before=(left.tobytes(),right.tobytes())
        canvas,info=comparison_canvas(left,right,scene_name="dewi_gangga",split="test",
            render_method="ours_30000",method_name="RaDe-GS",filename="00000.png")
        self.assertGreaterEqual(info["title_font_size"],60)
        self.assertGreater(info["header_height"],48)
        self.assertEqual(canvas.width,3840)
        file=BytesIO()
        canvas.save(file,format="PNG")
        file.seek(0)
        with Image.open(file) as saved:
            h=info["header_height"]
            self.assertEqual(saved.crop((0,h,1920,h+1080)).tobytes(),before[0])
            self.assertEqual(saved.crop((1920,h,3840,h+1080)).tobytes(),before[1])
        self.assertEqual((left.tobytes(),right.tobytes()),before)
        for line in info["labels"]:
            x0,y0,x1,y1=line["ink_box"]
            self.assertGreaterEqual(x0,line["panel"]*1920)
            self.assertLessEqual(x1,(line["panel"]+1)*1920)
            self.assertGreaterEqual(y0,0)
            self.assertLess(y1,info["header_height"])

    def test_long_method_names_and_filenames_wrap_without_covering_photos(self):
        width,height=320,90
        source=Image.new("RGB",(width,height),(8,19,30))
        canvas,info=comparison_canvas(source,source,scene_name="patung_dengan_nama_yang_panjang",
            split="test",render_method="ours_30000",
            method_name="RaDe-GS + Scaffold anchor initialization "*3,filename="nama_file_sangat_panjang_"*8+".png")
        for line in info["labels"]:
            x0,y0,x1,y1=line["ink_box"]
            self.assertGreaterEqual(x0,line["panel"]*width)
            self.assertLessEqual(x1,(line["panel"]+1)*width)
            self.assertGreaterEqual(y0,0)
            self.assertLess(y1,info["header_height"])
        h=info["header_height"]
        self.assertEqual(canvas.crop((0,h,width,h+height)).tobytes(),source.tobytes())
        self.assertEqual(canvas.crop((width,h,2*width,h+height)).tobytes(),source.tobytes())


if __name__ == "__main__":
    unittest.main()
