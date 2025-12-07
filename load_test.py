from locust import HttpUser, task

class LoadTest(HttpUser):
    @task
    def rec(self):
        self.client.post("/recommend", json={"user_id": 10, "k": 10})
