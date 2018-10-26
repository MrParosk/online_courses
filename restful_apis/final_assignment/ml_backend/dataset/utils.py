from ml_backend.dataset.models import DataPoint


def query_data():
    points = DataPoint.query.all()

    points_list = []
    for point in points:
        d = {"class_label": point.class_label,
             "sepal_length": point.sepal_length,
             "sepal_width": point.sepal_width,
             "pental_length": point.pental_length,
             "pental_width": point.pental_width
        }
        points_list.append(d)

    return points_list
