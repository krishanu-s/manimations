import numpy as np
import manim as m

class MovingZoomedSceneAround(m.ZoomedScene):
# contributed by TheoremofBeethoven, www.youtube.com/c/TheoremofBeethoven
    def __init__(self, **kwargs):
        m.ZoomedScene.__init__(
            self,
            zoom_factor=0.3,
            zoomed_display_height=1,
            zoomed_display_width=6,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
                },
            **kwargs
        )

    def construct(self):
        # dot = m.Dot().shift(m.UL * 2)
        time = m.ValueTracker(0)
        image = m.ImageMobject(np.random.randint(0, 256, (100, 100)))
        # image = m.ImageMobject(np.uint8([[0, 100, 30, 200],
        #                                [255, 0, 5, 33]]))
        image.height = 100

        self.add(image)
        self.play(time.animate.increment_value(5.0), run_time=5.0, rate_func=m.linear)
        # zoomed_camera = self.zoomed_camera
        # zoomed_display = self.zoomed_display
        # frame = zoomed_camera.frame
        # zoomed_display_frame = zoomed_display.display_frame

        # frame.move_to(dot)
        # frame.set_color(m.PURPLE)
        # zoomed_display_frame.set_color(m.RED)
        # zoomed_display.shift(m.DOWN)

        # zd_rect = m.BackgroundRectangle(zoomed_display, fill_opacity=0, buff=m.MED_SMALL_BUFF)
        # self.add_foreground_mobject(zd_rect)

        # unfold_camera = m.UpdateFromFunc(zd_rect, lambda rect: rect.replace(zoomed_display))

        # frame_text.next_to(frame, m.DOWN)

        # self.play(Create(frame), FadeIn(frame_text, shift=UP))
        # self.activate_zooming()

        # self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera)
        # zoomed_camera_text.next_to(zoomed_display_frame, DOWN)
        # self.play(FadeIn(zoomed_camera_text, shift=UP))
        # # Scale in        x   y  z
        # scale_factor = [0.5, 1.5, 0]
        # self.play(
        #     frame.animate.scale(scale_factor),
        #     zoomed_display.animate.scale(scale_factor),
        #     FadeOut(zoomed_camera_text),
        #     FadeOut(frame_text)
        # )
        # self.wait()
        # self.play(ScaleInPlace(zoomed_display, 2))
        # self.wait()
        # self.play(frame.animate.shift(2.5 * DOWN))
        # self.wait()
        # self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera, rate_func=lambda t: smooth(1 - t))
        # self.play(Uncreate(zoomed_display_frame), FadeOut(frame))
        # self.wait()



# don't remove below command for run button to work
# %manim -qm -v WARNING MovingZoomedSceneAround