Streaming and chunked input
===========================

Two independent kinds of streaming: frames arriving over time (decoder side)
and score matrices arriving in chunks (decodable side).

Incremental decoding
--------------------

``init_decoding`` + ``advance_decoding`` replace the one-shot ``decode``:

.. code-block:: python

   decoder = kd.LatticeFasterDecoder(graph, cfg)
   decoder.init_decoding()
   for chunk_of_frames in audio_stream:
       feed_scores(chunk_of_frames)          # into your decodable
       decoder.advance_decoding(decodable)   # consume all available frames
       print("frames decoded:", decoder.num_frames_decoded())
   decoder.finalize_decoding()               # optional; prunes final links
   ok, best = decoder.get_best_path()

``advance_decoding(decodable, max_num_frames=n)`` consumes at most ``n``
frames per call.  The decoder only looks at frames in
``[num_frames_decoded(), decodable.num_frames_ready())``, so making more
frames available in the decodable is what advances it.

Chunked score matrices
----------------------

:class:`~kaldi_decoder.DecodableMatrixMappedOffset` accepts score chunks and
discards old frames to bound memory:

.. code-block:: python

   dec = kd.DecodableMatrixMappedOffset(tm)
   for chunk in score_chunks:            # each (T x num_pdfs) float32
       dec.accept_loglikes(chunk, frames_to_discard=0)
       decoder.advance_decoding(dec)
       # optionally drop the frames the decoder has finished with:
       dec.accept_loglikes(next_chunk, frames_to_discard=frames_consumed)

   dec.input_is_finished()               # now is_last_frame() can be True

* ``frames_to_discard`` drops the OLDEST frames when the new chunk arrives.
* ``first_available_frame()`` tells you which frames are still queryable --
  querying older frames is an error (they are gone).
* ``input_is_finished()`` must be called before ``is_last_frame`` can return
  True -- without it the decoder cannot tell "stream paused" from "stream
  done".

Alternative: ``DecodableCtc(log_probs, offset=n)`` / ``DecodableMatrixMapped``
express the same idea by constructing a fresh object per chunk with a frame
offset.  That is simpler to reason about (no mutation) but re-allocates.
