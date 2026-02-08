def realtime_pipeline(signal_stream, model):
    """Placeholder real-time pipeline."""
    for signal in signal_stream:
        yield model(signal)
