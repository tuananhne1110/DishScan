class Camera:
    def __init__(self, id, url):
        self.id = id
        self.url = url
        self.zones = list()

    def get_info(self):
        return {
            "id": self.id,
            "url": self.url 
        }