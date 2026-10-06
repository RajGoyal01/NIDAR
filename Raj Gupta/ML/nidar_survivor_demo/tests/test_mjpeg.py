import unittest

from nidar_survivor_demo.camera.mjpeg_capture import MultipartJpegs


def part(body):
    return (b"\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
            str(len(body)).encode() + b"\r\n\r\n" + body)


class MultipartTests(unittest.TestCase):
    def test_split_headers_and_body(self):
        parser = MultipartJpegs()
        data = part(b"jpeg-payload")
        frames = []
        for byte in data:
            result = parser.feed(bytes([byte]))
            if result is not None:
                frames.append(result)
        self.assertEqual(frames, [b"jpeg-payload"])

    def test_only_latest_from_burst_and_partial_next(self):
        parser = MultipartJpegs()
        third = part(b"three")
        self.assertEqual(parser.feed(part(b"one") + part(b"two") + third[:-2]), b"two")
        self.assertEqual(parser.feed(third[-2:]), b"three")

    def test_bad_headers_and_size_limits(self):
        for data in (b"x" * 16385,
                     b"Content-Type: image/jpeg\r\n\r\n",
                     b"Content-Type: image/jpeg\r\nContent-Length: 8388609\r\n\r\n",
                     b"Content-Type: text/html\r\nContent-Length: 2\r\n\r\nhi"):
            with self.subTest(data=data[:60]), self.assertRaises(ValueError):
                MultipartJpegs().feed(data)
