"""run the flask app on twisted"""
# pylint: disable=invalid-name
# pylint: disable=no-member
from twisted.internet import reactor
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.util import redirectTo
from twisted.web.wsgi import WSGIResource
import pem
from catstream import app

catstream_site = WSGIResource(reactor, reactor.getThreadPool(), app)

ctxFactory = pem.twisted.certificateOptionsFromFiles(
    'key.pem', 'cert_and_chain.pem')
# thanks Let's Encrypt and ZeroSSL!
reactor.listenSSL(443, Site(catstream_site), ctxFactory)

# redirect http traffic to https
class HttpsRedirector(Resource):
    """redirect traffic to https"""
    def render(self, request):
        url_path = request.URLPath()
        url_path.scheme = 'https'
        return redirectTo(bytes(str(url_path), 'utf-8'), request)
    def getChild(self, path, request):
        return self
reactor.listenTCP(80, Site(HttpsRedirector()))

reactor.run()
