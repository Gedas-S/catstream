"""run the flask app on twisted"""
# pylint: disable=invalid-name
# pylint: disable=no-member
from twisted.internet import reactor
from twisted.web.server import Site
from twisted.web.wsgi import WSGIResource
from catstream import app

catstream_site = WSGIResource(reactor, reactor.getThreadPool(), app)

reactor.listenTCP(80, Site(catstream_site))
reactor.run()
