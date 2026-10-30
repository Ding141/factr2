from w3_clock_stability_probe import source_timing_ok


def test_batched_delivery_is_eligible_when_original_samples_are_contiguous_and_fresh():
    # Samples issued every 10 ms can arrive in a 50 ms burst without losing a
    # source sample. This remains inside the unchanged 250 ms freshness limit.
    assert source_timing_ok({'stream':.01},{'stream':.05},['stream'])


def test_missing_source_sample_stale_frame_and_missing_stream_fail():
    assert not source_timing_ok({'stream':.05},{'stream':.05},['stream'])
    assert not source_timing_ok({'stream':.01},{'stream':.30},['stream'])
    assert not source_timing_ok({}, {}, ['stream'])
