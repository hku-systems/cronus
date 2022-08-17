
#ifndef _BOOST_THREAD_THREAD_HPP
#define _BOOST_THREAD_THREAD_HPP

namespace boost {
    class thread {
        public:
            using id = int;
    };
    class this_thread {
        public:
            static thread::id get_id() { return 1; }
    };
    class mutex {
        public:
            void lock() {}
            void unlock() {}
    };
    class condition_variable {

    };
    
};


#endif