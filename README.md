# Data Visualization

This is a data visualization project that seizes the interactive power of `Bokeh` in order to visualize to models and see their different behaviors on a randomly generated dataset. This consists of two Bokeh Applications that run dashboards on the browser, which can be run as follows:

You can run both dashboards at once by navigating to the main directory and running: 
`bokeh serve --show analysis_dashboard_main.py management_dashboard_main.py`

Keep in mind that due to Bokeh some paths had to be added to sys.path in Python, which means that it might be necessary to change them according to your own sys.path. You do not need to run the generator, as it takes quite a while, everything's serialized and ready to go.