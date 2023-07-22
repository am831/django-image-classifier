from django.test import TestCase, Client

class TestViews():
    
    def test_home(self):
        client = Client()
        response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'homePage.html')

    def test_uploadBrain(self):
        client = Client()
        response = client.get('/uploadBrain')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifyBrain.html')

    def test_uploadSkin(self):
        client = Client()
        response = client.get('/uploadSkin')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifySkin.html')

    def test_predictImageBrain(self):
        client = Client()
        response = client.get('/predictImageBrain')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifyBrain.html')

    def test_predictImageSkin(self):
        client = Client()
        response = client.get('/predictImageSkin')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifySkin.html')

    def test_viewDataBase(self):
        client = Client()
        response = client.get('/viewDataBase')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'viewDatabase.html')

   
